from tqdm import tqdm
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from seqdiffusion import Diffusion
from models import Transformer, Critic, LSTMPolicy, RNNPolicy, GRUPolicy
from temporal import TemporalUnet
from datasets.normalization import DatasetNormalizer
torch.backends.cudnn.enabled = False
def generate_square_subsequent_mask(seq: torch.Tensor):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, *_ = seq.shape

    subsequent_mask = torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1).bool()
    return subsequent_mask

class Policy(object):
    def __init__(self,
                 env_name,
                 state_dim,
                 action_dim,
                 action_scale,
                 horizon,
                 device,
                 discount,
                 tau,
                 n_timesteps,
                 d_model=256,
                 d_ff=512,
                 use_attention=False,
                 dim_mults=(1, 2, 4, 8),
                 w=1,
                 history=None,
                 schedule = "linear",
                 lr=1e-4,
                 model='transformer') -> None:
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history = history
        if history is None:
            self.history = horizon - 1
        input_dim = state_dim
        self.min_action = float(action_scale.low[0])
        self.max_action = float(action_scale.high[0])
        self.mask = None

        print('use model: ', model)
        if model == 'transformer':
            n_heads, n_layers = 4, 4
            self.feasible_generator = Transformer(input_dim, d_model, n_heads, d_ff, n_layers, 0.1).to(device)
        elif model == 'lstm':
            self.feasible_generator = LSTMPolicy(state_dim, length=horizon).to(device)
        elif model == 'rnn':
            self.feasible_generator = RNNPolicy(state_dim, length=horizon).to(device)
        elif model == 'gru':
            self.feasible_generator = GRUPolicy(state_dim, length=horizon).to(device)
        self.feasible_generator_optimizer = torch.optim.Adam(self.feasible_generator.parameters(), lr=lr)

        #=====================================================================#
        #============================= Diffuser ==============================#
        #=====================================================================#
        self.model = TemporalUnet(horizon, state_dim, None, attention=use_attention, dim_mults=dim_mults).to(device)
        self.planner = Diffusion(state_dim, self.model, None, horizon=horizon, n_timesteps=n_timesteps, predict_epsilon=True, beta_schedule=schedule, w=w).to(device)# predict_epsilon=False
        self.planner_optimizer = torch.optim.AdamW(self.planner.parameters(), lr=lr, weight_decay=1e-4)
        self.ema = EMA(1-tau)
        self.ema_model = copy.deepcopy(self.planner)
        self.update_ema_every = 2

        #=====================================================================#
        #=============================== Actor ===============================#
        #=====================================================================#
        hidden_dim = 256
        self.actor = nn.Sequential(
                nn.Linear(2 * self.state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        #=====================================================================#
        #=============================== Critic ==============================#
        #=====================================================================#
        self.critic = Critic(state_dim, action_dim, length = horizon).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)



        self.device = device
        self.discount = discount
        self.tau = tau
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_timesteps = n_timesteps

        self.planner_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.planner_optimizer, 1, 1e-2, total_iters=1e5)
        self.feasible_generator_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.feasible_generator_optimizer, 1, 1e-2, total_iters=1e5)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.actor_optimizer, 1, 1e-2, total_iters=1e5)

        self.sqrt_alpha = self.planner.sqrt_alphas_cumprod[n_timesteps-1].unsqueeze(-1)
        self.sqrt_one_minus_alphas_cumprod = self.planner.sqrt_one_minus_alphas_cumprod[n_timesteps-1].unsqueeze(-1)

    def eval(self):
        self.feasible_generator.eval()
        self.ema_model.eval()
        self.planner.eval()
        self.actor.eval()
        self.critic.eval()


    def evaluate(self, env, eval_episodes=10, normalizer:DatasetNormalizer=None, rtg=None, scale=1.0, use_diffusion=True,progress=False, use_ddim=False) :
        scores = []
        self.feasible_generator.eval()
        self.ema_model.eval()
        return_to_go = rtg
        from utils.rendering import MuJoCoRenderer
        # render = MuJoCoRenderer(self.env_name)
        for _ in range(eval_episodes):
            state, done = env.reset(), False
            
            history_state = []
            history_rtg = []
            episode_reward = 0
            rtg = return_to_go
            rtg = rtg / scale
            # min_rtg = rtg / 3
            total_step = 0

            while not done:
                if isinstance(state, dict):
                    state = state["observation"]
                state = normalizer.normalize(state, "observations")
                history_state.append(state)
                history_rtg.append(rtg)
                # queue: np.ndarray => torch [1, horizon, state_dim]
                _state: torch.Tensor = torch.tensor(np.stack(history_state, 0), dtype=torch.float32).unsqueeze(0).to(self.device)
                _rtg: torch.Tensor = torch.tensor(np.stack(history_rtg, 0), dtype=torch.float32).unsqueeze(0).to(self.device)
                action = self.select_action(_state, _rtg, use_diffusion, use_ddim, normalizer)
                state, reward, done, _ = env.step(action)
                rtg -= reward / scale
                # rtg = np.clip(rtg, min_rtg, None)
                episode_reward += reward
                if progress:
                    total_step += 1
                    print(f"                                                              ", end="\r")
                    print(f"steps: {total_step} -------- rewards: {episode_reward}", end="\r")
            unnormaled_state = normalizer.unnormalize(_state[:1].detach().cpu().data.numpy(), "observations")
            # render.composite(f"./reference/{self.env_name}_traj.png", unnormaled_state)
            if progress:
                print(f"reward: {episode_reward}, normalized_scores: {env.get_normalized_score(episode_reward)}, total_step: {total_step}")
            scores.append(episode_reward)
        self.feasible_generator.train()
        self.ema_model.train()

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)

        #normlize
        # normalized_scores = [env.get_normalized_score(s) for s in scores]
        # avg_normalized_score = env.get_normalized_score(avg_score)
        # std_normalized_score = np.std(normalized_scores)
        # max_normalized_score = np.max(normalized_scores)



        return {"reward/avg": avg_score,
                "reward/std": std_score,
                # "reward/avg_normalized": avg_normalized_score,
                # "reward/std_normalized": std_normalized_score,
                "reward/max": max_score,}
                # "reward/max_normalized": max_normalized_score}

    def render_reference(self, renderer, trajectories, state_mean, state_std):
        '''
            renders training points
        '''
        normed_observations = trajectories
        observations = normed_observations * state_std + state_mean

        savepath = os.path.join("./reference", f'_sample-reference.png')
        if not os.path.exists("./reference"):
            os.makedirs("./reference")
        renderer.composite(savepath, observations)

    def train(self, batch):
        # batch = [observations, next_observations, actions, rtg]
        # state = [batch_size, len, state_dim]

        observations, next_observations, actions, rtg = batch

        observations = observations.to(self.device)
        next_observations = next_observations.to(self.device)
        actions = actions.to(self.device)
        rtg = rtg.unsqueeze(-1).to(self.device)

        states = observations[:, :-1]
        next_states = observations[:, 1:]
        cond = {0:observations[:,0]}

        # feasible generator
        state_pred, reward_pred = self.feasible_generator(states, rtg[:,:-1])
        t_loss = F.mse_loss(state_pred, next_states) + F.mse_loss(reward_pred, rtg[:, 1:])

        self.feasible_generator_optimizer.zero_grad()
        t_loss.backward()
        self.feasible_generator_optimizer.step()

        # planner
        p_loss:torch.Tensor = self.planner.loss(observations, cond, rtg[:, 0])

        self.planner_optimizer.zero_grad()
        p_loss.backward()
        self.planner_optimizer.step()

        # actor
        n_ns_pair = torch.cat([observations, next_observations], dim=-1).view(-1, self.state_dim*2)
        actions = actions.view(-1, self.action_dim)
        pred_actions = self.actor(n_ns_pair).clamp(self.min_action, self.max_action)
        a_loss = F.mse_loss(pred_actions, actions)

        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        # critic
        value = self.critic(observations.reshape(-1, self.state_dim), actions.reshape(-1, self.action_dim))
        rtg = rtg.contiguous().view(-1, 1)
        c_loss = F.mse_loss(value, rtg)

        self.critic_optimizer.zero_grad()
        c_loss.backward()
        self.critic_optimizer.step()


        self.planner_lr_scheduler.step()
        self.feasible_generator_lr_scheduler.step()
        self.actor_lr_scheduler.step()


        return {"loss/planner": p_loss.item(),
                "loss/transformer": t_loss.item(),
                "loss/critic": c_loss.item(),
                "loss/actor": a_loss.item(),
                }
    
    def save(self, path=None):
        if path is None:
            path = "./model/checkpoint.pth"
        prefix = os.path.dirname(path)
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        torch.save({
            'planner': self.planner.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'transformer': self.feasible_generator.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

    def load(self, path=None):
        if path is None:
            path = "./model/checkpoint.pth"
        checkpoint = torch.load(path, map_location=self.device)
        self.planner.load_state_dict(checkpoint['planner'])
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.feasible_generator.load_state_dict(checkpoint['transformer'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

    @torch.no_grad()
    def get_goal(self, observations, rtgs):
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(np.stack(observations, 0), dtype=torch.float32, device=self.device).unsqueeze(0)
            rtgs = torch.tensor(np.stack(rtgs, 0), dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)

        observations = observations[:, -self.history:]
        rtgs = rtgs[:, -self.history:]
        # cond = {0:observations[:,-1]}
        # condtion = rtgs[:, -1]

        length = 8
        for _ in range(length - 1):
            pred_state, pred_rtg = self.transformer(observations[:,-self.history:], rtgs[:,-self.history:])
            observations = torch.cat([observations, pred_state[:, -1:]], dim=1)
            rtgs = torch.cat([rtgs, pred_rtg[:,-1:]], dim=1)

        observations = observations[:, -length:, :]
        next_obs, goal = observations[:,1], observations[:, -1]
        return next_obs.squeeze().cpu().data.numpy(), goal.squeeze().cpu().data.numpy()


    def select_action(self, state, rtg, use_diffusion=True, use_ddim=False, normalizer=None):
        repeat = 32
        state = torch.repeat_interleave(state, repeat, dim=0)
        rtg = torch.repeat_interleave(rtg, repeat, dim=0).unsqueeze(-1)
        length = state.shape[1]
        cond = {0:state[:,-1]}
        condtion = rtg[:, -1]
        from utils.rendering import MuJoCoRenderer
        # render = MuJoCoRenderer(self.env_name)
        with torch.no_grad():
            if not use_ddim:
                for _ in range(self.horizon - 1):
                    # pred_state, pred_rtg = self.transformer(state[:,-self.history:], rtg[:,-self.history:])
                    pred_state, pred_rtg = self.feasible_generator(state[:,-self.history:], rtg[:,-self.history:])
                    state = torch.cat([state, pred_state[:, -1:]], dim=1)
                    rtg = torch.cat([rtg, pred_rtg[:,-1:]], dim=1)

                state = state[:, -self.horizon:, :]
                # if length % 100 == 0:
                #     unnormaled_state = normalizer.unnormalize(state[:1].detach().cpu().data.numpy(), "observations")
                #     render.composite(f"./reference/{self.env_name}_sample-reference_pre.png", unnormaled_state)
                state = torch.randn_like(state)
                state = self.ema_model.ddim_sample(state, cond, condtion, ddim_timesteps=5)
                # if length % 100 == 0:
                #     unnormaled_state = normalizer.unnormalize(state[:1].detach().cpu().data.numpy(), "observations")
                #     render.composite(f"./reference/{self.env_name}_sample-reference_after.png", unnormaled_state)
                if use_diffusion:
                    _noise_state = state
                    state = self.ema_model(_noise_state, cond, condtion)
                    # if length % 100 == 0:
                    # state = torch.cat([_noise_state, _state], dim=0)
            else:
                # self.ema_model.n_timesteps = 10
                state = torch.randn(state.shape[0], self.horizon, self.state_dim).to(self.device)
                state = self.ema_model.ddim_sample(state, cond, condtion, ddim_timesteps=10)
                # unnormaled_state = normalizer.unnormalize(state[:1].detach().cpu().data.numpy(), "observations")
                # render.composite(f"./reference/{self.env_name}_sample-reference.png", unnormaled_state)
                
            
            _cond = state[:, :2, :]
            assert not torch.isnan(_cond).any(), f"state: {_cond}"
            actions = self.actor(_cond.contiguous().view(-1, self.state_dim * 2)).squeeze()
            assert not torch.isnan(actions).any(), f"actions: {actions}"
            reward = self.critic(_cond[:, 0], actions).flatten()
            # not inf nan
            assert not torch.isnan(reward).any(), f"reward: {reward}"
            idx = torch.multinomial(F.softmax(reward, dim=0), num_samples=1)
            action = actions[idx].squeeze()
        return action.cpu().data.numpy().flatten()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.planner.state_dict())

    def step_ema(self, step, step_start_ema):
        if step < step_start_ema:
            self.reset_parameters()
            return
        if step % self.update_ema_every == 0:
            self.ema.update_model_average(self.ema_model, self.planner)


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new