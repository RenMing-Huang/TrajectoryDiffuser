from typing import Dict, List, Tuple, Union
import os
import shutil
import argparse
import tqdm
import gym
import numpy as np
import torch
import d4rl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from policy import Policy
from datasets.dataset import SequenceDatasetV2
from datasets.normalization import DatasetNormalizer
from helpers import cycle
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

hyperparameters = {
    'halfcheetah-medium-expert-v2':  {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 12000.0},
    'halfcheetah-medium-replay-v2':  {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5300.0},
    'halfcheetah-medium-v2':         {'lr': 2e-4, 'horizon': 8, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5300.0},
    'hopper-medium-expert-v2':       {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3700.0},
    'hopper-medium-replay-v2':       {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3100.0},
    'hopper-medium-v2':              {'lr': 2e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.2, 'rtg': 3100.0},
    'walker2d-medium-expert-v2':     {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 5100.0},
    'walker2d-medium-replay-v2':     {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 4200.0},
    'walker2d-medium-v2':            {'lr': 2e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 4200.0},
    'pen-human-v1':                  {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 2, 'scalar': 1.3, 'rtg': 6000.0},
    'pen-cloned-v1':                 {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 2, 'scalar': 1.3, 'rtg': 6000.0},
    'pen-expert-v1':                 {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 2, 'scalar': 1.3, 'rtg': 6000.0},
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
    'maze2d-umaze-v1':               {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 200.0},
    'maze2d-medium-v1':              {'lr': 1e-4, 'horizon': 64, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 300.0},
    'maze2d-large-v1':               {'lr': 1e-4, 'horizon': 128, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 400.0},
    'antmaze-umaze-v2':              {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0},
    'antmaze-large-play-v2':         {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0},
    'antmaze-large-diverse-v2':      {'lr': 1e-4, 'horizon': 32, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0},
    'FetchPush-v1':                  {'lr': 1e-4, 'horizon': 16, 'n_timesteps': 5, 'scalar': 1.1, 'rtg': 1.0, 'load_path':'/data3/hrenming/Trajectory_Diffuser/data/hard_tasks_2e5/expert_small/FetchPush'},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='hopper-medium-expert-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_freq', type=int, default=5e3)
    parser.add_argument('--save_freq', type=int, default=2e4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.005)
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

    horizon = hyperparameters[env_name]['horizon']
    n_timesteps = hyperparameters[env_name]['n_timesteps']
    lr = hyperparameters[env_name]['lr']
    w = hyperparameters[env_name]['scalar']
    rtg = hyperparameters[env_name]['rtg']
    try:
        load_path = hyperparameters[env_name]['load_path']
    except:
        load_path = None

    env = gym.make(env_name)
    device =  torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    if 'Fetch' in env_name:
        observation_dim = env.observation_space['observation'].shape[0] 
    else:
        observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = env.action_space

    scale = 1000
    if 'maze2d' in args.env_name:
        scale = 500
    elif 'halfcheetah' in args.env_name:
        scale = 10000
    elif 'antmaze' in args.env_name:
        scale = 1

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
                    lr=lr,
                    schedule=schedule,
                    model=model)
    
    dataset = SequenceDatasetV2(env_name,
                                horizon=horizon, 
                                returns_scale=scale,
                                termination_penalty=None,
                                load_path=load_path)
    normalizer:DatasetNormalizer = dataset.normalizer
    step_start_ema = 10000
    cnt = 0
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    totle_iteration = 2_000_000
    if os.path.exists(f"./runs/{env_name}_{model}"):
        # remove all files in the folder
        shutil.rmtree(f"./runs/{env_name}_{model}")

    writer = SummaryWriter(f"./runs/{env_name}_{model}")

    print("=================TRAINING START=================")
    for _, batch in enumerate(cycle(dataloader)):
        cnt += 1
        loss: Dict = policy.train(batch)
        policy.step_ema(cnt, step_start_ema)

        record(writer, "loss", loss, cnt)
        if cnt % eval_freq == 0:
            reward_td = policy.evaluate(env, 30, normalizer, rtg, scale, False)
            record(writer, "reward_td", reward_td, cnt)
            formate_print(loss, reward_td, cnt)
        if cnt == 1 or cnt % save_freq == 0:
            policy.save(f"./models/{env_name}_{model}/{cnt}.pth")
        if cnt == totle_iteration:
            break

def formate_print(loss:dict, reward:dict, cnt):
    print("========================================")
    print(f"iteration: {cnt}")
    for key in loss:
        print(f"{key}: {loss[key]}")
    print("----------------------------------------")
    for key in reward:
        print(f"{key}: {reward[key]}")
    print("========================================")

def record(writer, prefix, scalar, gloabl_step):
    for key in scalar:
        writer.add_scalar(f"{prefix}/{key}", scalar[key], gloabl_step)

if __name__ == "__main__":
    main()