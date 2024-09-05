import os
import torch
import datetime
import gym
import argparse
import d4rl
from datasets.dataset import SequenceDatasetV2
from datasets.normalization import DatasetNormalizer

import numpy as np
import random
# from utils.rendering import MuJoCoRenderer
from policy import Policy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
hyperparameters = {
    'halfcheetah-medium-expert-v2':  {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 12000.0},
    'halfcheetah-medium-replay-v2':  {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5300.0},
    'halfcheetah-medium-v2':         {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5300.0},
    'hopper-medium-expert-v2':       {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3600.0},
    'hopper-medium-replay-v2':       {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3100.0},
    'hopper-medium-v2':              {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3100.0},
    'walker2d-medium-expert-v2':     {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5100.0},
    'walker2d-medium-replay-v2':     {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5100.0},
    'walker2d-medium-v2':            {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 4200.0},
    'pen-human-v1':                  {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.3, 'rtg': 6000.0},
    'pen-cloned-v1':                 {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.3, 'rtg': 6000.0},
    'pen-expert-v1':                 {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.3, 'rtg': 6000.0},
    'kitchen-partial-v0':            {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 500.0},
    'kitchen-mixed-v0':              {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 400.0},
    'door-human-v0':                 {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 1500.0},
    'door-cloned-v0':                {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 1500.0},
    'door-expert-v0':                {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 1500.0},
    'hammer-human-v0':               {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 17000.0},
    'hammer-cloned-v0':              {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 17000.0},
    'hammer-expert-v0':              {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 17000.0},
    'relocate-human-v0':             {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 4000.0},
    'relocate-cloned-v0':            {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 4000.0},
    'relocate-expert-v0':            {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 4000.0},
    'maze2d-umaze-v1':               {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 300.0},
    'maze2d-medium-v1':              {'lr': 1e-4, 'horizon': 64, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 400.0},
    'maze2d-large-v1':               {'lr': 1e-4, 'horizon': 128, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 400.0},
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='halfcheetah-medium-expert-v2')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_freq', type=int, default=5e3)
    parser.add_argument('--save_freq', type=int, default=2e4)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--load_path', type=str, default='/data3/hrenming/Trajectory_Diffuser/models/halfcheetah-medium-expert-v2_lstm/260000.pth')
    parser.add_argument('--model', type=str, default='transformer')
    return parser.parse_args()


def main():
    # set_seed(114514)
    args = parse_args()
    gamma = args.gamma
    schedule = args.schedule
    eval_freq = args.eval_freq
    save_freq = args.save_freq
    env_name = args.env_name
    tau = args.tau
    model = args.model

    print(f"start evaluation: {env_name}")
    horizon = hyperparameters[env_name]['horizon']
    n_timesteps = hyperparameters[env_name]['n_timesteps']
    lr = hyperparameters[env_name]['lr']
    w = hyperparameters[env_name]['scalar']
    rtg = hyperparameters[env_name]['rtg']

    env = gym.make(env_name)
    device =  torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    action_scale = env.action_space

    scale = 1000
    if 'maze2d' in args.env_name:
        scale = 500
    if 'halfcheetah' in args.env_name:
        scale = 10000

    policy = Policy(args.env_name,
                    observation_dim,
                    action_dim,
                    action_scale,
                    horizon,
                    device,
                    w=w,
                    discount=gamma,
                    tau=tau,
                    n_timesteps=n_timesteps,
                    use_attention=False,
                    history=None,
                    lr=lr,
                    schedule=schedule,
                    model=model,)
    policy.load(args.load_path)
    
    dataset = SequenceDatasetV2(env_name,
                                horizon=horizon, 
                                returns_scale=scale,
                                termination_penalty=-100)
    normalizer:DatasetNormalizer = dataset.normalizer

    #sample trajectories
    print("sample trajectories")
    batch = dataset.__getitem__(0)
    trajectories = batch[0]
    rtg = batch[3]
    
    from utils.rendering import MuJoCoRenderer
    render = MuJoCoRenderer(args.env_name)
    # render_reference(render, trajectories, normalizer)
    render_samples(render, trajectories, normalizer, policy, rtg)

    # print("start evaluate")
    # policy.w = w
    # policy.history = 1
    # unupdated_reward = policy.evaluate(env, 10, state_mean, state_std, totle_reward, scale, update_rtg=False, test_mode=True, sample=True,renderer=render)
    # print(f"unupdated_reward: {unupdated_reward}")


def render_reference(renderer, trajectories, state_mean, state_std):
        '''
            renders training points
        '''
        normed_observations = trajectories
        observations = normed_observations * state_std + state_mean

        savepath = os.path.join("./reference", f'_sample-reference.png')
        if not os.path.exists("./reference"):
            os.makedirs("./reference")
        renderer.composite(savepath, observations)

def render_samples(renderer, trajectories:np.ndarray, normalizer, policy:Policy, rtg):
        '''
            renders samples from (ema) diffusion model
        '''
        
        print("rward is", rtg)
        if not os.path.exists("./reference"):
            os.makedirs("./reference")
        
        trajectories:torch.Tensor = torch.tensor(trajectories).to(policy.device)
        trajectories = normalizer.normalize(trajectories, 'observations')
        from copy import deepcopy
        trajectories_ = deepcopy(trajectories)
        cond = {0:trajectories[:, 0]}
        trajectories = trajectories * policy.sqrt_alpha + torch.randn_like(trajectories) * policy.sqrt_one_minus_alphas_cumprod
        # normed_observations = trajectories
        
        # savepath = os.path.join("./reference", f'_sample-reference_noise.png')
        # renderer.composite(savepath, observations)

        rtg = rtg[:,0].unsqueeze(1)
        trajectories = policy.ema_model(trajectories, cond, rtg)
        normed_observations = trajectories.cpu().detach().numpy()
        observations = normalizer.unnormalize(normed_observations, 'observations')
        savepath = os.path.join("./reference", f'_sample-reference_ema.png')
        renderer.composite(savepath, observations)

        rtg = torch.tensor([[0.2]]).to(policy.device).repeat(trajectories.shape[0], 1)
        trajectories = trajectories_
        source = trajectories[:, 0].unsqueeze(1)
        # source = torch.cat([torch.zeros([source.shape[0], policy.seq_length//2-1, source.shape[-1]], device=source.device), source], dim=1)
        for _ in range(policy.seq_length - 1):
            pred = policy.transformer(source[:, -10:], rtg)[:,-1:]
            source = torch.cat([source, pred], dim=1)
        
        normed_observations = source.cpu().detach().numpy()
        observations = normalizer.unnormalize(normed_observations, 'observations')
        savepath = os.path.join("./reference", f'_sample-reference_pred.png')
        renderer.composite(savepath, observations)


        env = gym.make(policy.env_name)
        state = env.reset()
        state = (state - state_mean) / state_std
        
        state = torch.tensor(state,dtype=torch.float32).to(policy.device).unsqueeze(0).unsqueeze(1)
        cond = {0:state[:, 0]}
        rtg = torch.tensor([[0.1]]).to(policy.device)
        for _ in range(policy.seq_length - 1):
            pred = policy.transformer(state[:, -1:], rtg)[:,-1:]
            state = torch.cat([state, pred], dim=1)
        normed_observations = state.cpu().detach().numpy()
        observations = normed_observations * state_std + state_mean
        savepath = os.path.join("./reference", f'_sample-reference_pred_init.png')
        renderer.composite(savepath, observations)

        cond[policy.seq_length-1] = state[:, -1]
        _state = torch.randn_like(state)
        trajectories = policy.ema_model(_state, cond, rtg)
        normed_observations = trajectories.cpu().detach().numpy()
        observations = normed_observations * state_std + state_mean
        savepath = os.path.join("./reference", f'_sample-reference_pred_ema.png')
        renderer.composite(savepath, observations)

        trajectories = state

        _cond1 = trajectories[:,:-1].reshape(-1, trajectories.shape[-1])
        _cond2 = trajectories[:,1:].reshape(-1, trajectories.shape[-1])
        _cond = torch.cat([_cond1, _cond2], dim=-1)
        actions = policy.actor(_cond)
        reward = policy.critic.q_min(_cond1, actions).flatten()
        print("reward is", reward)
        actions = actions.squeeze().cpu().detach().numpy()
        print("actions is", actions)
        rewards = 0
        pred_state = []
        for i in range(policy.seq_length - 1):
            next_state, reward, done, _ = env.step(actions[i])
            next_state = (next_state - state_mean) / state_std
            pred_state.append(next_state)
            rewards += reward
        print("get reward:", rewards)
        #save pred state
        pred_state = np.stack(pred_state)
        pred_state = pred_state.reshape(1, 15, -1)
        normed_observations = pred_state
        observations = normed_observations * state_std + state_mean
        savepath = os.path.join("./reference", f'_sample-reference_pred_ema2.png')
        renderer.composite(savepath, observations)


        # normed_observations = trajectories.cpu().detach().numpy()
        # observations = normed_observations * state_std + state_mean
        # savepath = os.path.join("./reference", f'_sample-reference_pred_noise_ema.png')
        # renderer.composite(savepath, observations)
        # source = torch.randn_like(source)
        # trajectories = policy.ema_model(source, cond, rtg)
        # normed_observations = trajectories.cpu().detach().numpy()
        # observations = normed_observations * state_std + state_mean
        # savepath = os.path.join("./reference", f'_sample-reference_noise_ema.png')
        # renderer.composite(savepath, observations)

if __name__ == "__main__":
    main()
