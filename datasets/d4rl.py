import os
import collections
import numpy as np
import gym
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    dataset = env.get_dataset()

    if 'antmaze' in str(env).lower():
        dataset = d4rl.qlearning_dataset(env)
        # antmaze: terminals are incorrect for GCRL
        dones_float = np.zeros_like(dataset['rewards'])
        dataset['terminals'][:] = 0.

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6:
                dataset['terminals'][i] = 1
            else:
                dataset['terminals'][i] = 0
        dataset['terminals'][-1] = 1

    return dataset

def load_dataset(path):
    keys = ['state', 'action', 'reward', 'not_done', 'next_state', ]
    path_list = [os.path.join(path, f'{key}.npy') for key in keys]
    dataset = dict()
    for key, path in zip(keys, path_list):
        dataset[key] = np.load(path).squeeze()
        if 'state' in key or 'next_state' in key:
            # For Fetch 3:6 is the achieved goal == -3:0
            dataset[key] = dataset[key][:,:-6]

    # rename keys
    dataset['observations'] = dataset.pop('state')
    dataset['actions'] = dataset.pop('action')
    dataset['rewards'] = dataset.pop('reward')
    dataset['terminals'] = 1-dataset.pop('not_done')
    dataset['next_observations'] = dataset.pop('next_state')
    return dataset


def sequence_dataset(env, preprocess_fn, load_path=None):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    if load_path is None:
        dataset = get_dataset(env)
        dataset = preprocess_fn(dataset)
    else:
        dataset = load_dataset(load_path)
        print(f'Loaded dataset from {load_path}')
        # show keys
        print(dataset.keys())

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env.max_episode_steps - 1)
            
            
        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:            
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name or 'pen' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
