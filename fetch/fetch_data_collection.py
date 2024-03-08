import sys
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision.transforms import Resize, CenterCrop, Grayscale
from PIL import Image
import argparse

from models.PPO import PPO







# for exploration_strategy in [34, 50, 78, 100, 126, 145, 176, 195]:
#     file_name = "./datasets_old/fetch_Push/" + "R=-" + str(exploration_strategy) + ".npz"
#     filenpz = np.load(file_name)
#     states, goals, actions, rewards, terminations, next_states = filenpz['arr_0'], filenpz['arr_1'], filenpz['arr_2'], filenpz['arr_3'], filenpz['arr_4'], filenpz['arr_5']
#
#     states = states[:, :25]
#     goals = goals[:, 25:]
#     next_states = next_states[:, :25]
#
#     file_name = "./datasets/fetch_Push/" + "R=-" + str(exploration_strategy) + ".npz"
#     np.savez(file_name, states, goals, actions, rewards, terminations, next_states)
#
#
# exit()
# for exploration_strategy in range(1, 51):
#     file_name = "./datasets_old/fetch_Reach/" + "R=-" + str(exploration_strategy) + ".npz"
#     filenpz = np.load(file_name)
#     states, goals, actions, rewards, terminations, next_states = filenpz['arr_0'], filenpz['arr_1'], filenpz['arr_2'], filenpz['arr_3'], filenpz['arr_4'], filenpz['arr_5']
#
#     goals = states[:, 10:]
#     states = states[:, :10]
#     next_states = next_states[:, :10]
#
#     file_name = "./datasets/fetch_Reach/" + "R=-" + str(exploration_strategy) + ".npz"
#     np.savez(file_name, states, goals, actions, rewards, terminations, next_states)
#
#
# exit()








# name Minari, name Gym, state space dim, goal space dim, action space dim
experiments = {0: {"name": "fetch_Reach", "gym_name": 'FetchReachDense-v2', "sim_name": 'FetchReach-v2', "state_dim": 10, "goal_dim": 3, "action_dim": 4},
               1: {"name": "fetch_Push", "gym_name": 'FetchPushDense-v2', "sim_name": 'FetchPush-v2', "state_dim": 25, "goal_dim": 3, "action_dim": 4}}

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--environment', default=1, type=int)
parser.add_argument('--reward', default=195, type=int)

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_flag = False

environment_details = experiments[args.environment]
input_dim, goal_dim, a_dim = environment_details["state_dim"], environment_details["goal_dim"], environment_details["action_dim"]


env_sparse = gym.make(environment_details["sim_name"], max_episode_steps=300)#, render_mode='human')

reward_file = args.reward
available_rewards = []
T = 200
N_TRJ = 1000

for reward_file in [34, 50, 78, 100, 126, 145, 176, 195]: #range(1, 51):

    try:
        specs = {'init_layer': False, 'norm': False, 'clip': False, 'mini_batch': False, 'learn_var': False}
        agent = PPO(input_dim + goal_dim, a_dim, 1, device, c2=0.0, K=10, specs=specs)
        reward_achieved = "-" + str(reward_file)
        path_dir = "./saved_PPOs/" + environment_details["name"] + "/"
        exp_name = "R=" + reward_achieved
        agent.load_model(path_dir, exp_name)
        # agent.policy.var *= 3
        # agent.old_policy.var *= 3

        agent_optimal = PPO(input_dim + goal_dim, a_dim, 1, device, c2=0.0, K=10, specs=specs)
        reward_achieved_opt = "-34" # "-1"
        exp_name_opt = "R=" + reward_achieved_opt
        agent_optimal.load_model(path_dir, exp_name_opt)
    except:
        continue

    available_rewards.append(reward_file)

    states, goals, actions, rewards, terminations, next_states = None, None, None, None, None, None

    epoch = 0
    avg_success = 0
    while epoch < N_TRJ:
        tmp_states, tmp_actions, tmp_rewards, tmp_terminations, tmp_next_states = [], [], [], [], []

        success = 0
        obs, _ = env_sparse.reset()
        for t in range(T):
            st = np.expand_dims(obs['observation'], 0)
            gt = np.expand_dims(obs['desired_goal'], 0)

            at, logprob, sigma = agent.get_action(torch.from_numpy(np.concatenate((st, gt), -1)).float().to(device), test=False)
            next_obs, reward_to_goal, terminated, truncated, info = env_sparse.step(at[0].detach().cpu().numpy())

            done = terminated or truncated

            success = np.maximum(success, reward_to_goal + 1)

            tmp_states.append(st)
            tmp_actions.append(at.detach().cpu().numpy())
            tmp_rewards.append(np.array([[reward_to_goal]]))
            tmp_terminations.append(np.array([[done * 1.]]))
            tmp_next_states.append(np.expand_dims(next_obs['observation'], 0))

            obs = next_obs
            if done:
                break

        tmp_terminations[-1][0, 0] = 1

        task_solved = False
        if reward_to_goal == 0:
            tmp_goal = np.expand_dims(next_obs['desired_goal'], 0)
            task_solved = True
        else:
            #print("GO OPTIMAL")
            for t in range(100):
                #env_sparse.render()
                st = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
                at, logprob, sigma = agent_optimal.get_action(torch.from_numpy(st).float().to(device), test=True)  # TODO test=True ???
                next_obs, reward_to_goal, terminated, truncated, info = env_sparse.step(at[0].detach().cpu().numpy())
                obs = next_obs
                if reward_to_goal == 0:
                    #print("SOLVED")
                    tmp_goal = np.expand_dims(next_obs['desired_goal'], 0)
                    task_solved = True
                    break

        if task_solved:
            if states is None:
                states = np.concatenate(tmp_states, 0)
                # goals = states * 0 + tmp_goal
                goals = np.repeat(tmp_goal, states.shape[0], axis=0)
                actions = np.concatenate(tmp_actions, 0)
                rewards = np.concatenate(tmp_rewards, 0)
                terminations = np.concatenate(tmp_terminations, 0)
                next_states = np.concatenate(tmp_next_states, 0)
            else:
                states = np.concatenate((states, np.concatenate(tmp_states, 0)), 0)
                # goals = np.concatenate((goals, np.concatenate(tmp_states, 0) * 0 + tmp_goal), 0)
                goals = np.concatenate((goals, np.repeat(tmp_goal, states.shape[0], axis=0)), 0)
                actions = np.concatenate((actions, np.concatenate(tmp_actions, 0)), 0)
                rewards = np.concatenate((rewards, np.concatenate(tmp_rewards, 0)), 0)
                terminations = np.concatenate((terminations, np.concatenate(tmp_terminations, 0)), 0)
                next_states = np.concatenate((next_states, np.concatenate(tmp_next_states, 0)), 0)

            epoch += 1
            avg_success += success

        sys.stdout.write("\rR="+str(reward_file)+" epoch %i" % epoch)
        sys.stdout.flush()

    print("---------------------")
    print("avg success: " + str(avg_success/N_TRJ))
    # file_name = "./datasets/fetch_Push_mix/" + exp_name + ".npz"
    file_name = "./datasets/" + environment_details["name"] + "/" + exp_name + ".npz"
    np.savez(file_name, states, goals, actions, rewards, terminations, next_states)


print()





















