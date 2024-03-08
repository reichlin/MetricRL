import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter

from models.DQN import DQN
from my_utils.model_utils import fill_up_actions, fill_up_transitions
from hyperMaze_env import HyperCube


parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--epsilon', default=0.5, type=float)
parser.add_argument('--curiosity_lambda', default=0.0, type=float)
parser.add_argument('--tau', default=0.1, type=float)

parser.add_argument('--cube_dim', default=4, type=int)
parser.add_argument('--cube_size', default=10, type=int)

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cpu')#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_flag = False

writer = SummaryWriter("./logs/debug_tau=0.1_eps=0.5_small")

all_actions, _ = fill_up_actions(np.zeros((3 ** args.cube_dim, args.cube_dim)), np.zeros(args.cube_dim), args.cube_dim, 0, 0)

input_dim, goal_dim, a_dim = args.cube_dim, args.cube_dim, all_actions.shape[0]


EPOCHS = 1000000
T = 300

env = HyperCube(dim=args.cube_dim, size=args.cube_size, max_T=T, use_obstacles=True, random_obstacles=False)

model = DQN(input_dim, a_dim, 0.9, 1000000, args.epsilon, args.tau, curiosity_lambda=0, n_layers=4, device=device).to(device)

for epoch in tqdm(range(EPOCHS)):

    model.train()

    total_reward = 0
    n_trj = 10
    for trj in range(n_trj):
        obs, _ = env.reset(random_reset=True)
        current_state = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)
        for t in range(100):
            a_idx = model.get_action(current_state, None, None)
            current_action = all_actions[a_idx]
            next_obs, reward, terminated, truncated, info = env.step(current_action)
            total_reward += reward
            next_state = torch.from_numpy(np.expand_dims(next_obs['observation'], 0)).float().to(device)
            model.replay_buffer.push(current_state, None, None, current_action, a_idx, reward, (not terminated)*1., next_state)
            current_state = next_state * 1.
            if terminated or truncated:
                break

    writer.add_scalar("Rewards/training", total_reward / n_trj, epoch)

    if len(model.replay_buffer) > 1000:
        for _ in range(10):
            log_losses, tr_r = model.update(batch_size=256)
        model.update_target(epoch)

        writer.add_scalar("Losses/DQN_loss", log_losses['L_DQN'], epoch)
        writer.add_scalar("Losses/Curiosity_error", log_losses['L_trans'], epoch)
        writer.add_scalar("Rewards/training", total_reward/n_trj, epoch)

    if epoch % 10 == 9:

        test_reward, tot_trials = 0, 10
        for _ in range(tot_trials):
            obs, _ = env.reset()
            current_state = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)

            for t in range(T):
                a_idx = model.get_action(current_state, None, None, test=True)
                current_action = all_actions[a_idx]
                next_obs, reward, terminated, truncated, info = env.step(current_action)
                test_reward += reward
                next_state = torch.from_numpy(np.expand_dims(next_obs['observation'], 0)).float().to(device)
                current_state = next_state * 1.
                if terminated or truncated:
                    break
        writer.add_scalar("Rewards/testing", test_reward / tot_trials, epoch)

print()

torch.save(model.state_dict(), "./saved_DQNs/dim="+str(args.cube_dim)+"_size="+str(args.cube_size)+".pt")

