import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import d3rlpy
import yaml

from models.metricRL import MetricRL
from models.quasimetric import QuasiMetric
from models.contrastiveRL import ContrastiveRL

from my_utils.model_utils import *
from my_utils.torch_datasets import Dataset_Custom
from my_utils.d3rlpy_utils import get_d3rlpy_algo
from my_utils.train_utils import simulation, train_loop

from miniGrid_envs import MultiGoalEnv
from minigrid.wrappers import FullyObsWrapper


def get_goal(goal):
    return goal



if os.getcwd().split('/')[-1] == 'miniGrid':
    root_folder = './'
elif os.getcwd().split('/')[-1] == 'cluster_sh':
    root_folder = '../miniGrid/'
else:
    root_folder = './miniGrid/'

with open('../config.yml', 'r') as file:
    configs = yaml.safe_load(file)

device = torch.device('cpu') #'cuda:0' if torch.cuda.is_available() else 'cpu')
device_flag = False


input_dim, goal_dim, a_dim = 2, 2, 4
z_dim = 128
reg = 1.0
batch_size = 256
batches_per_epoch = 200

policy_def = {'type': 'discrete', 'var': 1.0, 'max_action': 1.0, 'bias_action': 0.0}

results = np.zeros((5, 100))
for i_goal, n_goals in enumerate([2, 5, 10, 100, 1000]):

    N_TRJ = 1000
    EPOCHS = 100
    T = 100
    size = 50
    #n_goals = 2
    gamma = 0.95

    env = MultiGoalEnv(size=size, agent_start_pos=(1, 1), number_of_goals=n_goals, max_steps=T)
    env = FullyObsWrapper(env)

    states, goals, actions, rewards, terminations, next_states = None, None, None, None, None, None
    for i in tqdm(range(N_TRJ)):

        tmp_states, tmp_goals, tmp_actions, tmp_rewards, tmp_terminations, tmp_next_states = [], [], [], [], [], []

        obs, goal = env.reset(options={'random_pos': True})

        current_state = obs['state']

        for t in range(20):

            current_action = np.random.randint(0, 4)

            next_obs, reward, terminated, truncated, info = env.step(current_action)

            done = terminated or truncated

            tmp_states.append(obs['state'])
            tmp_goals.append(np.expand_dims(goal, 0))
            tmp_next_states.append(next_obs['state'])
            tmp_actions.append(np.array([[current_action]]))
            tmp_rewards.append(np.array([[reward]]))
            tmp_terminations.append(np.array([[done * 1.]]))

            obs = next_obs

            if done:
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


    # print()

    # dataset = Dataset_Trajectories(states, goals, actions, rewards, terminations, next_states, device, batches_per_epoch * batch_size, preload_states=True, connectify=False)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    offline_dataset = (states, goals, actions, rewards, terminations, next_states)
    dataset = Dataset_Custom(offline_dataset, device, batches_per_epoch * batch_size, get_goal, images=False, gamma=-1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    agent = MetricRL(input_dim,
                     goal_dim,
                     z_dim,
                     a_dim,
                     reg=reg,
                     gamma=gamma,
                     policy_def=policy_def,
                     get_goal=get_goal,
                     use_images=False,
                     device=device).to(device)

    writer = SummaryWriter("./logs/multigoals_debug_stop_train_n_goals="+str(n_goals))

    logs_idx = 0
    for epoch in tqdm(range(EPOCHS)):

        agent.train()
        for loader_idx, batch in enumerate(dataloader):
            log_losses = agent.update(batch, train_modules=(1, 0))

            writer.add_scalar("Losses/positive_loss", log_losses['L_pos'], logs_idx)
            writer.add_scalar("Losses/negative_loss", log_losses['L_neg'], logs_idx)
            writer.add_scalar("Losses/actor_loss", log_losses['L_pi'], logs_idx)
            logs_idx += 1

        if epoch % 10 == 9:
            #agent.train_pi = True

            for gamma in [0.9, 0.95, 0.99, 0.999]:
                agent.gamma = gamma
                goals = torch.tensor([[[int(size * 0.4), int(size * 0.2)], [int(size * 0.7), int(size * 0.8)]]]).float().to(device)
                rewards = torch.tensor([[0.7, 1]]).float().to(device)
                values = np.zeros((size - 2, size - 2))
                for i in range(size - 2):
                    for j in range(size - 2):
                        ss = torch.tensor([[i + 1, j + 1]]).float().to(device)
                        values[i, j] = agent.get_multi_value(ss, goals, rewards).detach().cpu().item()
                plt.imshow((values + np.min(values)) / (np.max(values) + np.min(values)), cmap='Blues', vmin=0.0, vmax=1.0)
                plt.scatter(int(size * 0.2) - 1, int(size * 0.4) - 1, c='tab:orange', marker='*', s=150, alpha=0.8)
                plt.scatter(int(size * 0.8) - 1, int(size * 0.7) - 1, c='tab:orange', marker='*', s=150, alpha=0.8)
                plt.axis('off')
                plt.show()


print()














