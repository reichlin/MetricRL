import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import minari
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--expl', default=0, type=int)  # 0, 1, 2
parser.add_argument('--env', default=0, type=int)  # 0, 1, 2
args = parser.parse_args()

exploration_processes = {0: "Random",
                         1: "Ornstein-Uhlenbeck",
                         2: "Minari"}

# name Minari, name Gym, state space dim, goal space dim, action space dim
experiments = {0: {"name": "point_uMaze", "minari_name": 'pointmaze-umaze-v1', "gym_name": 'PointMaze_UMaze-v3', "state_dim": 4, "goal_dim": 2, "action_dim": 2},
               1: {"name": "point_Medium", "minari_name": 'pointmaze-medium-v1', "gym_name": 'PointMaze_Medium_Diverse_GR-v3', "state_dim": 4, "goal_dim": 2, "action_dim": 2},
               2: {"name": "point_Large", "minari_name": 'pointmaze-large-v1', "gym_name": 'PointMaze_Large_Diverse_GR-v3', "state_dim": 4, "goal_dim": 2, "action_dim": 2}}

expl = args.expl
env_type = args.env

for expl in [0, 1]:
    for env_type in [0, 1, 2]:

        theta = 0.1
        sigma = 0.2
        exploration_strategy = exploration_processes[expl]

        # minari.download_dataset(dataset_id=experiments[env]["minari_name"])

        max_T = 1000000
        env = gym.make(experiments[env_type]["gym_name"], continuing_task=False)
        total_episodes = 10000

        states, goals, actions, rewards, terminations, next_states = None, None, None, None, None, None

        for _ in tqdm(range(total_episodes)):
            tmp_states, tmp_goals, tmp_actions, tmp_rewards, tmp_terminations, tmp_next_states = [], [], [], [], [], []
            obs, _ = env.reset()

            st = np.expand_dims(obs['observation'], 0)
            gt = np.expand_dims(obs['desired_goal'], 0)
            at = env.action_space.sample()
            for t in range(max_T):
                if exploration_strategy == "Ornstein-Uhlenbeck":
                    zt = env.action_space.sample()
                    at = (1-theta)*at + sigma*zt
                else:
                    at = env.action_space.sample()
                obs, r, term, trunc, _ = env.step(at)
                tmp_states.append(st)
                tmp_goals.append(gt)
                tmp_actions.append(np.expand_dims(at, 0))
                tmp_rewards.append(np.array([[r]]))
                tmp_terminations.append(np.array([[(term or trunc) * 1.]]))

                st = np.expand_dims(obs['observation'], 0)
                tmp_next_states.append(st)
                if term or trunc:
                    break

            tmp_states = np.concatenate(tmp_states, 0)
            tmp_goals = np.concatenate(tmp_goals, 0)
            tmp_actions = np.concatenate(tmp_actions, 0)
            tmp_rewards = np.concatenate(tmp_rewards, 0)
            tmp_terminations = np.concatenate(tmp_terminations, 0)
            tmp_next_states = np.concatenate(tmp_next_states, 0)
            if states is None:
                states = tmp_states
                goals = tmp_goals
                actions = tmp_actions
                rewards = tmp_rewards
                terminations = tmp_terminations
                next_states = tmp_next_states
            else:
                states = np.concatenate((states, tmp_states), 0)
                goals = np.concatenate((goals, tmp_goals), 0)
                actions = np.concatenate((actions, tmp_actions), 0)
                rewards = np.concatenate((rewards, tmp_rewards), 0)
                terminations = np.concatenate((terminations, tmp_terminations), 0)
                next_states = np.concatenate((next_states, tmp_next_states), 0)

            terminations[-1, 0] = 1.

        file_name = "./datasets/" + exploration_strategy + "_" + experiments[env_type]["name"] + ".npz"
        np.savez(file_name, states, goals, actions, rewards, terminations, next_states)







