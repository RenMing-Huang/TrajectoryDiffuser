from collections import namedtuple
import numpy as np
import torch
import d4rl

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

RewardBatch = namedtuple('Batch', 'observations next_observations actions rtg')
Batch = namedtuple('Batch', 'observations next_observations actions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

# def sequence_dataset(env, data):
#     """
#     Returns an iterator through trajectories.
#     Args:
#         env: An OfflineEnv object.
#         dataset: An optional dataset to pass in for processing. If None,
#             the dataset will default to env.get_dataset()
#         **kwargs: Arguments to pass to env.get_dataset().
#     Returns:
#         An iterator through dictionaries with keys:
#             observations
#             actions
#             rewards
#             terminals
#     """
#     dataset = data

#     N = dataset['rewards'].shape[0]
#     data_ = collections.defaultdict(list)

#     # The newer version of the dataset adds an explicit
#     # timeouts field. Keep old method for backwards compatability.
#     use_timeouts = 'timeouts' in dataset
#     if not use_timeouts:
#         dataset['timeouts'] = np.zeros_like(dataset['terminals'])
#     episode_step = 0
#     for i in range(N):
#         done_bool = bool(dataset['terminals'][i])
#         if use_timeouts:
#             final_timestep = dataset['timeouts'][i]
#         else:
#             final_timestep = (episode_step == env._max_episode_steps - 1)
#             dataset['timeouts'][i] = final_timestep
    
#         for k in dataset:
#             if 'metadata' in k: continue
#             data_[k].append(dataset[k][i])

#         if done_bool or final_timestep:
#             episode_step = 0
#             episode_data = {}
#             for k in data_:
#                 episode_data[k] = np.array(data_[k])
#             yield episode_data
#             data_ = collections.defaultdict(list)

#         episode_step += 1

class SequenceDatasetV2(torch.utils.data.Dataset):
    def __init__(self, 
                env_name,
                normalizer='GaussianNormalizer',
                use_normalizer=True,
                preprocess_fns=[], 
                horizon=16, 
                max_n_episodes=20000, 
                termination_penalty=0, 
                use_padding=False, 
                discount=1, 
                returns_scale=1000, 
                include_returns=True,
                load_path=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env_name)
        self.env = env = load_environment(env_name)
        # get the max length of env
        self.max_path_length = 1000
        self.horizon = horizon
        self.returns_scale = returns_scale
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length, dtype=np.float32)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn, load_path=load_path)

        fields = ReplayBuffer(max_n_episodes, self.max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        if use_normalizer:
            self.normalize(keys=["observations", "next_observations"])

        print(fields)

    def re_make_indices(self):
        self.indices = self.make_indices(self.fields.path_lengths, self.horizon)

    def normalize(self, keys):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            for j in range(path_length - horizon):
                indices.append((i, j, j + horizon))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        length = self.path_lengths[path_ind]
        observations = self.fields.normed_observations[path_ind, start:end]
        next_observations = self.fields.normed_next_observations[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]


        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:length]
            discounts = self.discounts[:len(rewards)]
            returns = np.cumsum(discounts * rewards[::-1])[::-1][:self.horizon] / self.returns_scale
            return observations, next_observations, actions, returns

        return observations, next_observations, actions


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, itr, horizon, state_mean, state_std, scale,device, penalty=None) -> None:
        super().__init__()

        self.device = device
        self.horizon = horizon
        self.termination_penalty = penalty

        self.observations = []
        self.actions = []
        self.next_observations = []
        self.rewards = []
        self.rtg = []
        self.terminals = []
        self.reward_to_go = []
        discount = 1


        for i, episode in enumerate(itr):
            # reward to go
            rtg = []

            episode_return = 0.0

            # penalty last step
            if self.termination_penalty is not None and episode['terminals'][-1] and not episode['timeouts'][-1]:
                episode['rewards'][-1] += self.termination_penalty


            episode["rewards"] = episode["rewards"] / scale
            for i in range(len(episode["rewards"])-1,-1,-1):
                episode_return = episode["rewards"][i] + episode_return * discount
                rtg.append(episode_return)

            rtg.reverse()

            # self.max_reward = max(self.max_reward, rtg[0])

            for j in range(0, len(episode["observations"]) - horizon + 1, 1):
                self.observations.append(episode["observations"][j:j+horizon])
                self.actions.append(episode["actions"][j:j+horizon])
                self.next_observations.append(episode["next_observations"][j:j+horizon])
                self.rewards.append(rtg[j:j+horizon])
                self.terminals.append(episode["terminals"][j:j+horizon])
                self.reward_to_go.append(rtg[0])



        # normalize
        self.observations = (np.array(self.observations) - state_mean) / (state_std + 1e-8)
        self.next_observations = (np.array(self.next_observations) - state_mean) / (state_std + 1e-8)

        # to tensor
        self.observations = torch.from_numpy(self.observations).float()
        self.actions = torch.tensor(np.array(self.actions), dtype=torch.float32)
        self.next_observations = torch.tensor(np.array(self.next_observations), dtype=torch.float32)
        self.rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32)
        self.terminals = torch.tensor(np.array(self.terminals), dtype=torch.float32)
        self.reward_to_go = torch.tensor(np.array(self.reward_to_go), dtype=torch.float32)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):
        return (
            self.observations[index],
            self.actions[index],
            self.next_observations[index],
            self.rewards[index],
            self.terminals[index],
            self.reward_to_go[index],
            
        )

def iql_normalize(reward, not_done):
	trajs_rt = []
	episode_return = 0.0
	for i in range(len(reward)):
		episode_return += reward[i]
		if not not_done[i]:
			trajs_rt.append(episode_return)
			episode_return = 0.0
	rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
	reward /= (rt_max - rt_min)
	reward *= 1000.
	return reward

class NormalDataset(torch.utils.data.Dataset):
        def __init__(self, data, device, reward_tune='normalize'):
            state = torch.from_numpy(data['observations']).float()
            action = torch.from_numpy(data['actions']).float()
            next_state = torch.from_numpy(data['next_observations']).float()
            #normalize
            self.state_mean, self.state_std = state.mean(dim=0), state.std(dim=0)
            # self.action_mean, self.action_std = action.mean(dim=0), action.std(dim=0)
            self.state = (state - state.mean(dim=0)) / state.std(dim=0)
            self.next_state = (next_state - state.mean(dim=0)) / state.std(dim=0)
            self.action = action

            
            reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
            self.done = torch.from_numpy(data['terminals']).view(-1, 1).float()
            self.size = self.state.shape[0]
            self.state_dim = self.state.shape[1]
            self.action_dim = self.action.shape[1]

            self.device = device

            if reward_tune == 'normalize':
                reward = (reward - reward.mean()) / reward.std()
            elif reward_tune == 'iql_antmaze':
                reward = reward - 1.0
            elif reward_tune == 'iql_locomotion':
                reward = iql_normalize(reward, self.not_done)
            elif reward_tune == 'cql_antmaze':
                reward = (reward - 0.5) * 4.0
            elif reward_tune == 'antmaze':
                reward = (reward - 0.25) * 2.0
            self.reward = reward
        def __len__(self):
            return self.size
        
        def __getitem__(self, ind):

            return (
                self.state[ind].to(self.device),
                self.action[ind].to(self.device),
                self.next_state[ind].to(self.device),
                self.reward[ind].to(self.device),
                self.done[ind].to(self.device)
            )

        def get_states(self):
            return self.state_mean, self.state_std