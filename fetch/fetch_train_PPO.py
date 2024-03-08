import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse

from models.PPO import PPO



experiments = {0: {"name": "fetch_Reach", "gym_name": 'FetchReachDense-v2', "sim_name": 'FetchReach-v2', "state_dim": 10, "goal_dim": 3, "action_dim": 4},
               1: {"name": "fetch_Push", "gym_name": 'FetchPushDense-v2', "sim_name": 'FetchPush-v2', "state_dim": 25, "goal_dim": 3, "action_dim": 4}}

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--environment', default=0, type=int)

parser.add_argument('--r2', default=0.1, type=float)
parser.add_argument('--c2', default=0.01, type=float)
parser.add_argument('--K', default=10, type=int)
parser.add_argument('--init_layer', default=0, type=int)
parser.add_argument('--norm', default=0, type=int)
parser.add_argument('--clip', default=0, type=int)
parser.add_argument('--mini_batch', default=0, type=int)
parser.add_argument('--learn_var', default=1, type=int)

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cpu')

r2 = args.r2
c2 = args.c2
K = args.K
T = 200

specs = {'init_layer': args.init_layer==1, 'norm': args.norm==1, 'clip': args.clip==1, 'mini_batch': args.mini_batch==1, 'learn_var': args.learn_var==1}

environment_details = experiments[args.environment]
input_dim, goal_dim, a_dim = environment_details["state_dim"], environment_details["goal_dim"], environment_details["action_dim"]

env_dense = gym.make(environment_details["gym_name"], max_episode_steps=T)#, render_mode='human')
env_sparse = gym.make(environment_details["sim_name"], max_episode_steps=T)

writer = SummaryWriter("./logs_PPO/" + environment_details["name"] + "_r2="+str(r2)+"_c2="+str(c2)+"_K="+str(K) + "_all_tricks=" + str(specs) + "_pushing_point_T=200_all_relative_bigger_max_var")


agent = PPO(input_dim+goal_dim, a_dim, 1, device, c2=c2, K=K, specs=specs)

EPOCHS = 10000000000000
n_trj = 10
test_trj = 10

for epoch in tqdm(range(EPOCHS)):
    avg_distance_to_cube, avg_distance_to_goal = 0, 0

    for i in range(n_trj):
        obs, _ = env_dense.reset()
        for t in range(T):
            #env.render()
            current_state = torch.from_numpy(np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)).float().to(device)

            at, logprob, sigma = agent.get_action(current_state)
            next_obs, reward_to_goal, terminated, truncated, info = env_dense.step(at[0].detach().cpu().numpy())

            training_reward = -(np.linalg.norm(next_obs['desired_goal'][:3] - next_obs['achieved_goal'][:3]) - \
                                np.linalg.norm(obs['desired_goal'][:3] - obs['achieved_goal'][:3]))
            if args.environment == 1:
                p_goal_old = obs['desired_goal'][:3]
                p_cube_old = obs['achieved_goal'][:3]
                pushing_point_old = p_cube_old + ((p_cube_old - p_goal_old) / np.linalg.norm(p_cube_old - p_goal_old)) * 0.06
                p_goal = next_obs['desired_goal'][:3]
                p_cube = next_obs['achieved_goal'][:3]
                pushing_point = p_cube + ((p_cube - p_goal) / np.linalg.norm(p_cube - p_goal)) * 0.06
                training_reward += - r2 * (np.linalg.norm(pushing_point - next_obs['observation'][:3]) - np.linalg.norm(pushing_point_old - obs['observation'][:3]))


            avg_distance_to_cube += np.linalg.norm(next_obs['achieved_goal'] - next_obs['observation'][:3])
            avg_distance_to_goal += - reward_to_goal

            done = terminated or truncated
            agent.push_batchdata(current_state.detach().cpu(), at.detach().cpu(), logprob.detach().cpu(), training_reward, done)

            obs = next_obs

            if done:
                break

    v_loss, h_loss = agent.update()
    agent.clear_batchdata()

    writer.add_scalar("Loss/v_loss", v_loss, epoch)
    writer.add_scalar("Loss/h_loss", h_loss, epoch)
    writer.add_scalar("Rewards/distance_to_cube", avg_distance_to_cube/(n_trj*T), epoch)
    writer.add_scalar("Rewards/distance_to_goal", avg_distance_to_goal/(n_trj*T), epoch)

    if epoch % 10 == 9:
        avg_reward = 0
        for i in range(test_trj):
            tot_reward = 0
            obs, _ = env_sparse.reset()
            for t in range(T):
                current_state = torch.from_numpy(np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)).float().to(device)

                at, logprob, sigma = agent.get_action(current_state, test=True)
                next_obs, reward, terminated, truncated, info = env_sparse.step(at[0].detach().cpu().numpy())
                tot_reward += reward

                done = terminated or truncated
                obs = next_obs
                if done:
                    break
            avg_reward += tot_reward
        avg_reward /= test_trj
        writer.add_scalar("Rewards/test_sparse_reward", avg_reward, epoch)

        reward_achieved = str(int(avg_reward))
        path_dir = "./saved_PPOs/" + environment_details["name"] + "/"
        exp_name = "R="+reward_achieved
        agent.save_model(path_dir, exp_name)


writer.close()
























