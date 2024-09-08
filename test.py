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
    'halfcheetah-medium-expert-v2':  {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 12000.0},
    'halfcheetah-medium-replay-v2':  {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5300.0},
    'halfcheetah-medium-v2':         {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5300.0},
    'hopper-medium-expert-v2':       {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3600.0},
    'hopper-medium-replay-v2':       {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3100.0},
    'hopper-medium-v2':              {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3100.0},
    'walker2d-medium-expert-v2':     {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 4900.0},
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
    parser.add_argument('--env_name', type=str, default='walker2d-medium-expert-v2')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_freq', type=int, default=5e3)
    parser.add_argument('--save_freq', type=int, default=2e4)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--load_path', type=str, default='/data3/hrenming/Trajectory_Diffuser/models/walker2d-medium-expert-v2_lstm/160000.pth')
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
    # for w in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0]:
    #     print(f"start evaluation: w = {w}")
    #     policy.w = w
    #     reward = policy.evaluate(env, 5, normalizer, rtg, scale, use_diffusion=True)
    #     formate_print(reward)
    # for _rtg in [rtg, 0.75*rtg, 0.5*rtg]:
    #     print(f"start evaluation: rtg = {_rtg}")
    #     policy.w = w
    #     reward = policy.evaluate(env, 10, normalizer, _rtg, scale, use_diffusion=True)
    #     formate_print(reward)
    # for step in [20, 50, 100]:
    #     print("start evaluation: step = ", step)
    #     policy.n_timesteps = step
    #     reward = policy.evaluate(env, 20, normalizer, rtg, scale, use_diffusion=True, progress=False)
    #     formate_print(reward)
    # policy.ema_model.n_timesteps = 100
    reward = policy.evaluate(env, 10, normalizer, rtg, scale, use_diffusion=False, progress=True, use_ddim=False)
    formate_print(reward)
    # reward = policy.evaluate(env, 10, normalizer, rtg, scale, use_diffusion=False, progress=True)
    # formate_print(reward)
    


def formate_print(reward:dict):
    for key in reward:
        print(f"{key}: {reward[key]}")

if __name__ == "__main__":
    main()