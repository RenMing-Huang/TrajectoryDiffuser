
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import (cosine_beta_schedule,
                     linear_beta_schedule,
                     vp_beta_schedule,
                     extract,
                     Losses)
from utils.utils import Progress, Silent

def apply_condition(seq, cond):
    for key, value in cond.items():
        seq[:, key] = value.clone()
    return seq

class Diffusion(nn.Module):
    def __init__(self, state_dim, model, max_action, horizon=20,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=False, predict_epsilon=True, w=1):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.max_action = max_action
        self.horizon = horizon
        self.model = model
        self.w = w
        
        if beta_schedule == 'linear':
            self.max_steps = 100
            betas = linear_beta_schedule(100)
        elif beta_schedule == 'cosine':
            self.max_steps = 20
            betas = cosine_beta_schedule(20)
        elif beta_schedule == 'vp':
            self.max_steps = 10
            betas = vp_beta_schedule(10)

        self.gamma = [1 for _ in np.linspace(0, 1, horizon)]
        self.gamma = torch.tensor(self.gamma, dtype=torch.float32) # shape (horizon,)
        # betas shape (n_timesteps, 1)
        betas = betas.unsqueeze(1)
        betas = betas * self.gamma # shape (n_timesteps, horizon)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones((1, self.horizon)), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # math: Var[x_{t-1}] = Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        # return x0; from x_t = \sqrt(\bar{\alpha}_t) x0 + \sqrt(1 - \bar{\alpha}_t) \epsilon_t

        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        # caculate mean:u_q = (beta_t * sqrt(\bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t)) * x_start + \\
        #               (1 - \bar{\alpha}_{t-1}) * sqrt(\alpha_t) / (1 - \bar{\alpha}_t) * x_t
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, c):
        epsilon_cond = self.model(x, c, t, use_dropout=False)
        epsilon_uncond = self.model(x, c, t, use_dropout=False, force_dropout=True)
        epsilon = epsilon_uncond + self.w*(epsilon_cond - epsilon_uncond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, c):
        b, l, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, c=c)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # x_{t-1} = mean + std * noise
        # where std = sqrt(Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, x, cond, reward, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        # x = torch.randn(shape, device=device)
        x = apply_condition(x, cond)
        if return_diffusion: diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size, ), i, device=device, dtype=torch.long)

            x = self.p_sample(x, timesteps, reward)
            x = apply_condition(x, cond)
            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, x, cond, reward, *args, **kwargs):
        batch_size = x.shape[0]
        shape = (batch_size, self.horizon, self.state_dim)
        x = self.p_sample_loop(x, cond, reward, shape, *args, **kwargs)
        return x
    
    # ----------------------------------------- ddim sample ----------------------------------------#
    
    def ddim_sample(self, x, cond, reward, ddim_timesteps=20, ddim_discr_method="uniform", ddim_eta=0.1, clip_denoised=False):
        if ddim_discr_method == 'uniform':
            c = self.max_steps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.max_steps, c)))
        else:
            raise NotImplementedError()
        
        if clip_denoised:
            assert self.predict_epsilon, "clip_denoised=True requires predict_epsilon=True"
        
        ddim_timestep_seq = ddim_timestep_seq + 1
        # clip to max_steps
        # ddim_timestep_seq = np.clip(ddim_timestep_seq, 0, self.max_steps-1)
        
        # previous sequence
        # ddim_timestep_prev_seq = ddim_timestep_seq[:-1]
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq)

        batch_size = x.shape[0]
        device = self.betas.device
        x = apply_condition(x, cond)
        # x is pure noise
        for i in reversed(range(0, ddim_timesteps)):
            timesteps = torch.full((batch_size, ), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_timesteps = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_cumprod, timesteps, x.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_timesteps, x.shape)

            # 2. predict noise using model
            epsilon_cond = self.model(x, reward, timesteps, use_dropout=False)
            epsilon_uncond = self.model(x, reward, timesteps, use_dropout=False, force_dropout=True)
            pred_noise = epsilon_uncond + self.w*(epsilon_cond - epsilon_uncond)

            # 3. get the predicted x_0
            pred_x0 = (x - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x)

            x = x_prev
            x = apply_condition(x, cond)

        return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None, cond=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # math:
        # x_{t-1} = \sqrt{1 - \alpha_t} * x_0 + \sqrt{\alpha_t} * \epsilon_t
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, cond, reward, t, weights=1.0):
        
        noise = torch.randn_like(x_start)
        noise[:, 0] = 0.0
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, cond=cond)
        x_recon = self.model(x_noisy, reward, time=t)

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            x_recon = apply_condition(x_recon, cond)
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, cond=None, reward=None, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.max_steps, (batch_size, ), device=x.device).long()
        return self.p_losses(x, cond, reward, t, weights)

    def forward(self, x, cond, reward, *args, **kwargs):
        return self.sample(x, cond, reward, *args, **kwargs)


class Uncond_Diffusion(nn.Module):
    def __init__(self, state_dim, model, max_action, horizon=20,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=False, predict_epsilon=True):
        super(Uncond_Diffusion, self).__init__()

        self.state_dim = state_dim
        self.max_action = max_action
        self.horizon = horizon
        self.model = model

        if beta_schedule == 'linear':
            self.max_steps = 100
            betas = linear_beta_schedule(100)
        elif beta_schedule == 'cosine':
            self.max_steps = 20
            betas = cosine_beta_schedule(20)
        elif beta_schedule == 'vp':
            self.max_steps = 10
            betas = vp_beta_schedule(10)

        self.gamma = [1 for _ in np.linspace(0, 1, horizon)]
        self.gamma = torch.tensor(self.gamma, dtype=torch.float32) # shape (horizon,)
        # betas shape (n_timesteps, 1)
        betas = betas.unsqueeze(1)
        betas = betas * self.gamma # shape (n_timesteps, horizon)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones((1, self.horizon)), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # math: Var[x_{t-1}] = Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        # return x0; from x_t = \sqrt(\bar{\alpha}_t) x0 + \sqrt(1 - \bar{\alpha}_t) \epsilon_t

        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        # caculate mean:u_q = (beta_t * sqrt(\bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t)) * x_start + \\
        #               (1 - \bar{\alpha}_{t-1}) * sqrt(\alpha_t) / (1 - \bar{\alpha}_t) * x_t
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t):
        epsilon = self.model(x, t, use_dropout=False)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t):
        b, l, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # x_{t-1} = mean + std * noise
        # where std = sqrt(Var[x_t] * (1 - alpha_{t-1}) / (1 - alpha_t))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, x,  shape, t=None, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        # x = torch.randn(shape, device=device)
        if t is None:
            t = self.n_timesteps
        if return_diffusion: diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()
        print(t)
        for i in reversed(range(0, t)):
            timesteps = torch.full((batch_size, ), i, device=device, dtype=torch.long)

            x = self.p_sample(x, timesteps)
            
            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, x, *args, **kwargs):
        batch_size = x.shape[0]
        shape = (batch_size, self.horizon, self.state_dim)
        x = self.p_sample_loop(x, shape, *args, **kwargs)
        return x
    def t_sample(self, x, t, *args, **kwargs):
        batch_size = x.shape[0]
        shape = (batch_size, self.horizon, self.state_dim)
        x = self.p_sample_loop(x,  shape, t = t, *args, **kwargs)
        return x
    
    # ----------------------------------------- ddim sample ----------------------------------------#
    
    def ddim_sample(self, x, cond, reward, ddim_timesteps=20, ddim_discr_method="uniform", ddim_eta=0.1, clip_denoised=False):
        if ddim_discr_method == 'uniform':
            c = self.max_steps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.max_steps, c)))
        else:
            raise NotImplementedError()
        
        if clip_denoised:
            assert self.predict_epsilon, "clip_denoised=True requires predict_epsilon=True"
        
        ddim_timestep_seq = ddim_timestep_seq + 1
        # clip to max_steps
        # ddim_timestep_seq = np.clip(ddim_timestep_seq, 0, self.max_steps-1)
        
        # previous sequence
        # ddim_timestep_prev_seq = ddim_timestep_seq[:-1]
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq)

        batch_size = x.shape[0]
        device = self.betas.device
        x = apply_condition(x, cond)
        # x is pure noise
        for i in reversed(range(0, ddim_timesteps)):
            timesteps = torch.full((batch_size, ), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_timesteps = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_cumprod, timesteps, x.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_timesteps, x.shape)

            # 2. predict noise using model
            epsilon_cond = self.model(x, reward, timesteps, use_dropout=False)
            epsilon_uncond = self.model(x, reward, timesteps, use_dropout=False, force_dropout=True)
            pred_noise = epsilon_uncond + self.w*(epsilon_cond - epsilon_uncond)

            # 3. get the predicted x_0
            pred_x0 = (x - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x)

            x = x_prev
            x = apply_condition(x, cond)

        return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None, cond=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # math:
        # x_{t-1} = \sqrt{1 - \alpha_t} * x_0 + \sqrt{\alpha_t} * \epsilon_t
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, t, weights=1.0):
        
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, time=t)

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.max_steps, (batch_size, ), device=x.device).long()
        return self.p_losses(x, t, weights)

    def forward(self, x, t=None, *args, **kwargs):
        print(t)
        if t is not None:
            return self.t_sample(x, t, *args, **kwargs)
        return self.sample(x, *args, **kwargs)