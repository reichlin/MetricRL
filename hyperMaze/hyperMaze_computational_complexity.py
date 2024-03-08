import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse

from models.DQN import DQN
from models.metricRL import MetricRL
from my_utils.model_utils import *
from hyperMaze_env import HyperCube


def get_goal(goal):
    return goal



print("FINISH IMPLEMENTING ...")
exit()


parser = argparse.ArgumentParser()

parser.add_argument('--model_type', default=1, type=int)  # 0: MetricRL, 1: DQN

parser.add_argument('--cube_dim', default=2, type=int)
parser.add_argument('--cube_size', default=100, type=int)
parser.add_argument('--random_obstacles', default=0, type=int)
parser.add_argument('--data_collection', default=2, type=int)  # 0: random_offline, 1: random_online, 2: complete

parser.add_argument('--batch_size', default=128, type=int)
#parser.add_argument('--epsilon', default=1.0, type=float)
parser.add_argument('--tau', default=0.01, type=float)

parser.add_argument('--use_policy', default=0, type=int)
parser.add_argument('--z_dim', default=128, type=int)
parser.add_argument('--reg', default=10.0, type=float, help="contrastive negative")

parser.add_argument('--seed', default=0, type=int)

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

data_collection_types = {0: "random_offline", 1: "random_online", 2: "complete"}
data_collection = data_collection_types[args.data_collection]


EPOCHS = 1000000
sim_frq = 100
batch_size = args.batch_size
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dim_input = args.cube_dim
dim_z = args.z_dim
goal_dim = args.cube_dim

#epsilon = args.epsilon
gamma = 0.99#5
tau = args.tau
n_layers = 4
reg = args.reg
use_policy = args.use_policy

max_T = args.cube_size*10

name_exp = data_collection + "_cube_dim="+str(args.cube_dim)+"_cube_size="+str(args.cube_size)+"_rand_obst="+str(args.random_obstacles)+"_bs="+str(batch_size)#+"_eps="+str(epsilon)
if args.model_type == 0:
    name_exp = "MetricRL_" + name_exp + "_z_dim="+str(dim_z)+"_reg="+str(reg)+"_use_policy="+str(use_policy)
else:
    name_exp = "DQN_" + name_exp + "_tau=" + str(tau)
name_exp += "_seed="+str(seed)

writer = SummaryWriter("./logs_cube/"+name_exp+"")

if not os.path.isdir("../saved_results/Comp_Complex"):
    os.mkdir("../saved_results/Comp_Complex")

env = HyperCube(dim=args.cube_dim, size=args.cube_size, max_T=max_T, use_obstacles=True, random_obstacles=(args.random_obstacles==1))

if data_collection == "complete":
    all_transitions = fill_up_transitions(env, [], np.zeros(args.cube_dim), 0)
    memory_size = len(all_transitions)
else:
    memory_size = 100000

all_actions, _ = fill_up_actions(np.zeros((3 ** args.cube_dim, args.cube_dim)), np.zeros(args.cube_dim), args.cube_dim, 0, 0)
all_actions_idx = torch.arange(all_actions.shape[0]).float().to(device)


if args.model_type == 0:
    policy_def = {'type': 'discrete', 'var': 1.0, 'max_action': 1.0, 'bias_action': 0.0}
    model = MetricRL(dim_input,
                     goal_dim,
                     dim_z,
                     all_actions.shape[0],
                     memory_size=memory_size,
                     reg=reg,
                     policy_def=policy_def,
                     get_goal=get_goal,
                     device=device).to(device)
    # all_actions_idx=all_actions_idx, all_actions=all_actions, reg=reg
else:
    model = DQN(dim_input, all_actions.shape[0], gamma, memory_size, 0.5, tau, n_layers=n_layers, device=device).to(device)


if data_collection == "complete":
    for step in all_transitions:

        state = torch.from_numpy(np.expand_dims(step['s'], 0)).float().to(device)
        obst = torch.from_numpy(np.expand_dims(step['obst'], 0)).float().to(device)
        goal = torch.from_numpy(np.expand_dims(step['g'], 0)).float().to(device)
        next_state = torch.from_numpy(np.expand_dims(step['s1'], 0)).float().to(device)

        model.replay_buffer.push(state, obst, goal, step['a'], step['a_idx'], step['r'], step['term'], next_state)

elif data_collection == "random_offline":

    for epoch in tqdm(range(memory_size)):

        obs, _ = env.reset(random_reset=True)
        current_goal = torch.from_numpy(np.expand_dims(obs['desired_goal'], 0)).float().to(device)
        current_obst = torch.from_numpy(np.expand_dims(obs['obstacles'], 0)).float().to(device)
        current_state = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)
        for t in range(1):
            current_action, a_idx = env.get_random_actions()
            next_obs, reward, terminated, truncated, info = env.step(current_action)
            next_goal = torch.from_numpy(np.expand_dims(next_obs['desired_goal'], 0)).float().to(device)
            next_obst = torch.from_numpy(np.expand_dims(next_obs['obstacles'], 0)).float().to(device)
            next_state = torch.from_numpy(np.expand_dims(next_obs['observation'], 0)).float().to(device)
            model.replay_buffer.push(current_state, current_obst, current_goal, current_action, a_idx, reward, (not terminated)*1., next_state)
            current_goal = next_goal * 1.
            current_state = next_state * 1.
            current_obst = next_obst * 1.
            if terminated or truncated:
                break

n_times_solved = 0
saved_rewards_test = []
for epoch in tqdm(range(EPOCHS)):

    model.train()

    total_reward = None
    if data_collection == "random_online":

        total_reward = 0
        n_trj = 10
        for trj in range(n_trj):
            obs, _ = env.reset(random_reset=True)
            current_goal = torch.from_numpy(np.expand_dims(obs['desired_goal'], 0)).float().to(device)
            current_obst = torch.from_numpy(np.expand_dims(obs['obstacles'], 0)).float().to(device)
            current_state = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)
            for t in range(max_T):
                a_idx = model.get_action(current_state, current_obst, current_goal)
                current_action = all_actions[a_idx]
                next_obs, reward, terminated, truncated, info = env.step(current_action)
                total_reward += reward
                next_goal = torch.from_numpy(np.expand_dims(next_obs['desired_goal'], 0)).float().to(device)
                next_obst = torch.from_numpy(np.expand_dims(next_obs['obstacles'], 0)).float().to(device)
                next_state = torch.from_numpy(np.expand_dims(next_obs['observation'], 0)).float().to(device)
                model.replay_buffer.push(current_state, current_obst, current_goal, current_action, a_idx, reward, (not terminated)*1., next_state)
                current_goal = next_goal * 1.
                current_state = next_state * 1.
                current_obst = next_obst * 1.
                if terminated or truncated:
                    break

    log_losses = model.update(batch_size=batch_size)
    model.update_target(epoch)

    if log_losses is not None:
        writer.add_scalar("Losses/positive_loss", log_losses['L_pos'], epoch)
        writer.add_scalar("Losses/negative_loss", log_losses['L_neg'], epoch)
        writer.add_scalar("Losses/transition_loss", log_losses['L_trans'], epoch)
        writer.add_scalar("Losses/DQN_loss", log_losses['L_DQN'], epoch)
        if total_reward is not None:
            writer.add_scalar("Rewards/train", total_reward/n_trj, epoch)

    #
    if epoch % sim_frq == sim_frq-1:

        total_reward = 0
        trj = []
        model.eval()
        obs, _ = env.reset()
        current_goal = torch.from_numpy(np.expand_dims(obs['desired_goal'], 0)).float().to(device)
        current_obst = torch.from_numpy(np.expand_dims(obs['obstacles'], 0)).float().to(device)
        current_state = torch.from_numpy(np.expand_dims(obs['observation'], 0)).float().to(device)
        for t in range(max_T):

            trj.append(current_state.detach().cpu().numpy())

            a_idx = model.get_action(current_state, current_obst, current_goal, test=True)
            current_action = all_actions[a_idx]

            next_obs, reward, terminated, truncated, info = env.step(current_action)
            total_reward += reward
            next_goal = torch.from_numpy(np.expand_dims(next_obs['desired_goal'], 0)).float().to(device)
            next_obst = torch.from_numpy(np.expand_dims(next_obs['obstacles'], 0)).float().to(device)
            next_state = torch.from_numpy(np.expand_dims(next_obs['observation'], 0)).float().to(device)
            if terminated or truncated:
                break
            if (next_state == current_state).all():
                break
            current_goal = next_goal * 1.
            current_state = next_state * 1.
            current_obst = next_obst * 1.

        writer.add_scalar("Rewards/test", total_reward, int(epoch / sim_frq))

        if total_reward > 0:
            n_times_solved += 1
        else:
            n_times_solved = 0

        saved_rewards_test.append(total_reward)
        np.savez("../saved_results/Comp_Complex/" + name_exp + ".npz", np.array(saved_rewards_test))


        if args.cube_dim == 2:

            img_trj = np.zeros((args.cube_size, args.cube_size))
            if env.obstacles is not None:
                for obs in env.obstacles:
                    img_trj[int(obs[0] * args.cube_size), int(obs[1] * args.cube_size)] = 1
            for step in trj:
                img_trj[int(step[0, 0] * args.cube_size), int(step[0, 1] * args.cube_size)] = 2
            fig = plt.figure()
            plt.imshow(img_trj)
            writer.add_figure("trj", fig, int(epoch / sim_frq))

            img_v = np.zeros((args.cube_size, args.cube_size))
            for i in range(args.cube_size):
                for j in range(args.cube_size):
                    state = torch.tensor([[i, j]]).float().to(device) / args.cube_size
                    goal = torch.from_numpy(env.goal).view(1, -1).float().to(device)
                    v = model.get_value(state, goal)
                    img_v[i, j] = v
            max_v = np.max(img_v)
            min_v = np.min(img_v)
            diff_v = max_v - min_v
            if env.obstacles is not None:
                for obs in env.obstacles:
                    img_v[int(obs[0] * args.cube_size), int(obs[1] * args.cube_size)] = min_v - diff_v / 10.
            fig = plt.figure()
            plt.imshow(img_v)
            writer.add_figure("values", fig, int(epoch / sim_frq))

    if n_times_solved >= 100:
        break


writer.close()











