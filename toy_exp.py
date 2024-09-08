from typing import Dict, List, Tuple, Union
import os
import shutil
import argparse
import tqdm
import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

def generate_data(num, device='cpu'):
    high_freq_prob = 0.8
    low_freq_prob = 0.2

    high_freq_num, low_freq_num = int(num * high_freq_prob), int(num * low_freq_prob)

    high_freq_alpha = 5
    low_freq_alpha = 3

    trajectory_lenght = 64
    # f(x) = cos(\alpha * 2 * pi x)
    # generate_trajectory
    high_freq_x = np.linspace(0, 1, trajectory_lenght)
    low_freq_x = np.linspace(0, 1, trajectory_lenght)

    # expand to high_freq_num, low_freq_num
    high_freq_y = np.cos(high_freq_alpha * 2 * np.pi * high_freq_x) + np.random.normal(0, 0.1, (high_freq_num, trajectory_lenght))
    
    low_freq_y = np.cos(low_freq_alpha * 2 * np.pi * low_freq_x) + np.random.normal(0, 0.1, (low_freq_num, trajectory_lenght))

    return torch.tensor(np.concatenate([high_freq_y, low_freq_y], axis=0), dtype=torch.float32).to(device)

#set seed
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)


device = 'cuda:2'
trajectories = generate_data(500, device)
from scipy.fftpack import fft
fs = 64 / 2
half_x = np.linspace(0, fs, 32)
def fourier_analysis(samples):
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    samples = samples.squeeze()
    freq = fft(samples)
    abs_y = np.abs(freq)

    normalization_y=abs_y/32
    normalization_y = normalization_y[:32]

    return normalization_y

fig, axs = plt.subplots(2, 3, figsize=(5 * 4, 6))



# plot all trajectories
import matplotlib.pyplot as plt
x = np.linspace(0, 2, 64)

for i in range(trajectories.shape[0]):
    axs[0][0].plot(x, trajectories[i].cpu().numpy(), alpha=0.01, c='blue')
    axs[0][0].set_xlabel('x', fontsize=24)
    axs[0][0].set_ylabel('y', fontsize=24)
    axs[0][0].set_title('Train data', fontsize=24)
    freq = fourier_analysis(trajectories[i])
    axs[1][0].plot(half_x, freq, alpha=0.01, c='red')
    axs[1][0].set_xlabel('Frequency $f$/Hz', fontsize=24)
    axs[1][0].set_ylabel('Magnitude', fontsize=24)

# use diffusion to model this data
from torch import nn
from torch.nn import functional as F

from diffusion import Uncond_Diffusion as Diffusion
from temporal import UncondTemporalUnet as TemporalUnet


class Replaybuffer:
    def __init__(self, traj) -> None:
        self.traj = traj
    def sample(self, batch_size):
        indices = np.random.choice(self.traj.shape[0], batch_size)
        return self.traj[indices]

state_dim = 1

horizon = 64
n_timesteps = 100
replaybuffer = Replaybuffer(trajectories)
model = TemporalUnet(horizon, state_dim, dim=32).to(device)
planner = Diffusion(state_dim, model, None, horizon=horizon, n_timesteps=n_timesteps, predict_epsilon=True, beta_schedule='linear').to(device)# predict_epsilon=False

# train the model
optimizer = torch.optim.Adam(planner.parameters(), lr=1e-3)

for i in range(1000):
    batch = replaybuffer.sample(32)
    batch = batch.unsqueeze(-1)
    optimizer.zero_grad()
    loss = planner.loss(batch)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f'loss: {loss.item()}')


# sample N trajectories
N = 256

trajs = torch.randn(N, horizon, state_dim).to(device)
with torch.no_grad():
    samples = planner(trajs)

# plot 
# x = np.linspace(0, 2, 64)
# for i in range(N):
#     axs[0][1].plot(x, samples[i].cpu().numpy(), alpha=0.01, c='blue')
#     freq = fourier_analysis(samples[i])
#     axs[1][1].plot(half_x, freq, alpha=0.01, c='blue')


import models
import imp
imp.reload(models)
from models import UncondTransformer as Transformer
transfomer = Transformer(state_dim, 32, 2, 64, 2, 0.1).to(device)

optimizer = torch.optim.Adam(transfomer.parameters(), lr=1e-3)

for i in range(1000):
    batch = replaybuffer.sample(32)
    batch = batch.unsqueeze(-1)
    optimizer.zero_grad()
    input = batch[:, :-1]
    target = batch[:, 1:]
    output = transfomer(input)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f'loss: {loss.item()}')

state = torch.tensor([[1.0]]).repeat_interleave(N, dim=0).unsqueeze(-1).to(device) #+ torch.randn(N, 1, 1).to(device) * 0.2
print(state.shape)
transfomer.train()
with torch.no_grad():
    samples = transfomer.sample(state, 64)
# plot
x = np.linspace(0, 2, 64)
for i in range(N):
    axs[0][1].plot(x, samples[i].cpu().numpy(), alpha=0.01, c='blue')
    axs[0][1].set_xlabel('x', fontsize=24)
    axs[0][1].set_ylabel('y', fontsize=24)
    axs[0][1].set_title('transformer planning', fontsize=24)
    freq = fourier_analysis(samples[i])
    axs[1][1].plot(half_x, freq, alpha=0.01, c='red')
    axs[1][1].set_xlabel('Frequency $f$/Hz', fontsize=24)
    axs[1][1].set_ylabel('Magnitude', fontsize=24)

# optimize by diffusion
with torch.no_grad():
    samples = transfomer.sample(state, 64)
    samples = planner(samples, t=20)
    # samples = torch.randn(N, 64, 1).to(device)
    # samples = planner(samples)

# plot
x = np.linspace(0, 2, 64)
for i in range(N):
    axs[0][2].plot(x, samples[i].cpu().numpy(), alpha=0.01, c='blue')
    axs[0][2].set_xlabel('x', fontsize=24)
    axs[0][2].set_ylabel('y', fontsize=24)
    axs[0][2].set_title('Optimized by diffusion', fontsize=24)
    freq = fourier_analysis(samples[i])
    axs[1][2].plot(half_x, freq, alpha=0.01, c='red')
    axs[1][2].set_xlabel('Frequency $f$/Hz', fontsize=24)
    axs[1][2].set_ylabel('Magnitude', fontsize=24)

# save fig
plt.tight_layout()
fig.savefig('toy_exp.pdf')


# half_ys = []
# # sample frequency

# for sample in samples:
#     half_y = fourier_analysis(sample.cpu().numpy())
#     half_ys.append(half_y)
#     plt.plot(half_x,half_y , c='red', alpha=0.01)

# # plot mean
# # mean_y = np.mean(half_ys, axis=0)
# # plt.plot(half_x, mean_y, c='blue',)

# # histogram


# plt.show()

