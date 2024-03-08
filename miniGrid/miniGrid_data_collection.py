import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import argparse

from models.DQN import DQN

from miniGrid_envs import KeyEnv, OpenEnv
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper


experiments = {0: {"name": "open_goal", "state_dim": 4, "goal_dim": 4, "action_dim": 4},
               1: {"name": "key_door_goal", "state_dim": 7, "goal_dim": 7, "action_dim": 6}}

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--environment', default=0, type=int)
parser.add_argument('--use_images', default=0, type=int)
parser.add_argument('--epsilon', default=0.0, type=float)

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for environment in [0, 1]:

    environment_details = experiments[environment]
    input_dim, goal_dim, a_dim = environment_details["state_dim"], environment_details["goal_dim"], environment_details["action_dim"]

    #use_images = args.use_images == 1


    for use_images in [False, True]:
        for epsilon in [0.1, 0.2, 0.5, 0.8, 0.9]:

            print(environment, use_images, epsilon)
            print()

            T = 100
            if use_images:
                if environment == 0:
                    env = OpenEnv(use_images=True, max_steps=T, size=10)  # render_mode="human"
                    env = RGBImgObsWrapper(env)  # get directly obs['image']
                else:
                    env = KeyEnv(use_images=True, max_steps=T)  # render_mode="human"
                    env = RGBImgObsWrapper(env)  # get directly obs['image']
            else:
                if environment == 0:
                    env = OpenEnv(use_images=False, max_steps=T, size=10)  # render_mode="human"
                    env = FullyObsWrapper(env)  # get directly obs['image']
                else:
                    env = KeyEnv(use_images=False, max_steps=T)  # render_mode="human"
                    env = FullyObsWrapper(env)  # get directly obs['image']


            model = DQN(input_dim, a_dim, 0.9, 1000000, epsilon, 0.1, device=device).to(device)
            if environment == 0:
                dqn_file_name = "./saved_DQNs/size=10_open_goal_eps=" + str(0.5) + "_cur=" + str(0.0) + "_tau=" + str(1.0) + ".pt"
            else:
                dqn_file_name = "./saved_DQNs/size=10_key_door_goal" + "_eps=" + str(0.75) + "_cur=" + str(0.0) + "_tau=" + str(0.1) + ".pt"
            model.load_state_dict(torch.load(dqn_file_name))
            model.eval()

            N_TRJ = 200#1000
            states, goals, actions, rewards, terminations, next_states = None, None, None, None, None, None
            for step in tqdm(range(N_TRJ)):

                tmp_states, tmp_goals, tmp_actions, tmp_rewards, tmp_terminations, tmp_next_states = [], [], [], [], [], []

                obs, goal = env.reset()
                current_state = torch.from_numpy(obs['state']).float().to(device)

                for t in range(T):

                    current_state = torch.from_numpy(obs['state']).float().to(device)

                    a_idx = model.get_action(current_state, None, None)
                    current_action = a_idx

                    next_obs, reward, terminated, truncated, info = env.step(current_action)

                    done = terminated or truncated

                    if use_images:
                        tmp_states.append(np.expand_dims(obs['image'], 0))
                        tmp_goals.append(np.expand_dims(goal, 0))
                        tmp_next_states.append(np.expand_dims(next_obs['image'], 0))
                    else:
                        tmp_states.append(obs['state'])
                        # goal_state = obs['state']
                        # goal_state[0, :2] = goal_state[0, -2:]
                        # if args.environment == 1:
                        #     goal_state[0, 2:4] = -1
                        # tmp_goals.append(goal_state)
                        tmp_goals.append(goal)
                        tmp_next_states.append(next_obs['state'])

                    tmp_actions.append(np.array([[current_action]]))
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

            file_name = "./datasets/"+environment_details['name']+"_img="+str(use_images)+"_eps="+str(epsilon)+".npz"
            np.savez(file_name, states, goals, actions, rewards, terminations, next_states)








