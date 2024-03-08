import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import argparse

from models.DQN import DQN
from my_utils.model_utils import fill_up_actions, fill_up_transitions
from hyper_cube_env import HyperCube



cube_dim = 4
cube_size = 10
tau = 0.1

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

all_actions, _ = fill_up_actions(np.zeros((3 ** cube_dim, cube_dim)), np.zeros(cube_dim), cube_dim, 0, 0)
input_dim, goal_dim, a_dim = cube_dim, cube_dim, all_actions.shape[0]

T = 25
env = HyperCube(dim=cube_dim, size=cube_size, max_T=T, use_obstacles=True, random_obstacles=False)


for epsilon in [0.0, 0.1, 0.2, 0.5, 0.8, 0.9]:

    print(epsilon)
    print()

    model = DQN(input_dim, a_dim, 0.9, 1000000, epsilon, tau, curiosity_lambda=0, n_layers=4, device=device).to(device)
    dqn_file_name = "./saved_DQNs/dim="+str(cube_dim)+"_size="+str(cube_size)+".pt"
    model.load_state_dict(torch.load(dqn_file_name))
    model.eval()

    N_TRJ = 1000
    states, goals, actions, rewards, terminations, next_states = None, None, None, None, None, None
    for step in tqdm(range(N_TRJ)):

        tmp_states, tmp_goals, tmp_actions, tmp_rewards, tmp_terminations, tmp_next_states = [], [], [], [], [], []

        obs, _ = env.reset(random_reset=True)
        goal = np.expand_dims(env.goal, 0)

        for t in range(T):

            current_state = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)

            a_idx = model.get_action(current_state, None, None)
            current_action = all_actions[a_idx]

            next_obs, reward, terminated, truncated, info = env.step(current_action)

            done = terminated or truncated

            tmp_states.append(np.expand_dims(obs['observation'], 0))
            tmp_goals.append(goal)
            tmp_next_states.append(np.expand_dims(next_obs['observation'], 0))

            tmp_actions.append(np.array([[a_idx]]))
            tmp_rewards.append(np.array([[reward]]))
            tmp_terminations.append(np.array([[done * 1.]]))

            obs = next_obs

            if terminated:
                break

        tmp_terminations[-1][0, 0] = 1

        if states is None:
            states = np.concatenate(tmp_states, 0)
            goals = np.concatenate(tmp_goals, 0)
            actions = np.concatenate(tmp_actions, 0)
            rewards = np.concatenate(tmp_rewards, 0)
            terminations = np.concatenate(tmp_terminations, 0)
            next_states = np.concatenate(tmp_next_states, 0)
        else:
            states = np.concatenate((states, np.concatenate(tmp_states, 0)), 0)
            goals = np.concatenate((goals, np.concatenate(tmp_goals, 0)), 0)
            actions = np.concatenate((actions, np.concatenate(tmp_actions, 0)), 0)
            rewards = np.concatenate((rewards, np.concatenate(tmp_rewards, 0)), 0)
            terminations = np.concatenate((terminations, np.concatenate(tmp_terminations, 0)), 0)
            next_states = np.concatenate((next_states, np.concatenate(tmp_next_states, 0)), 0)

    file_name = "./datasets/dim="+str(cube_dim)+"_size="+str(cube_size)+"_eps="+str(epsilon)+".npz"
    np.savez(file_name, states, goals, actions, rewards, terminations, next_states)








