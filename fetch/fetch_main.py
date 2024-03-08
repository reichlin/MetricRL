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
from my_utils.torch_datasets import Dataset_Custom, Dataset_Minari
from my_utils.d3rlpy_utils import get_d3rlpy_algo
from my_utils.train_utils import simulation, train_loop


def get_goal_reach(goal):
    if torch.is_tensor(goal):
        return torch.cat([goal, torch.zeros((goal.shape[0], 7)).float().to(goal.device)], -1)
    return np.concatenate((goal, np.zeros((goal.shape[0], 7))), -1)

def get_goal_push(goal):
    if torch.is_tensor(goal):
        return torch.cat([goal, goal, torch.zeros((goal.shape[0], 19)).float().to(goal.device)], -1)
    return np.concatenate((goal, goal, np.zeros((goal.shape[0], 19))), -1)



if os.getcwd().split('/')[-1] == 'fetch':
    exp_folder = './'
    root_folder = '../'
elif os.getcwd().split('/')[-1] == 'cluster_sh':
    exp_folder = '../fetch/'
    root_folder = '../'
else:
    exp_folder = './fetch/'
    root_folder = './'

with open(root_folder+'config.yml', 'r') as file:
    configs = yaml.safe_load(file)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
''' Dataset and Algorithm '''
parser.add_argument('--exploration', default=45, type=int)
parser.add_argument('--environment', default=0, type=int)
parser.add_argument('--algorithm', default=0, type=int)
''' MetricRL HyperParameters '''
parser.add_argument('--z_dim', default=128, type=int)
parser.add_argument('--reg', default=1.0, type=float, help="contrastive negative")
''' Baselines HyperParameters '''
parser.add_argument('--n_critics', default=2, type=int)
parser.add_argument('--n_actions', default=10, type=int)
parser.add_argument('--conservative_weight', default=1.0, type=float)
parser.add_argument('--expectile', default=0.7, type=float)
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_flag = False

''' HyperParameters '''
reward_norm = configs['fetch']["reward_norm"]
algo_id = args.algorithm
algo_name = configs["algorithms"][algo_id]
exploration_strategy = args.exploration
environment_details = configs['fetch']["experiments"][args.environment]
EPOCHS, batches_per_epoch, batch_size, gamma = configs["EPOCHS"], configs["batches_per_epoch"], configs["batch_size"], configs["gamma"]
input_dim, goal_dim, a_dim = environment_details["state_dim"], environment_details["goal_dim"], environment_details["action_dim"]

z_dim = args.z_dim
reg = args.reg

get_goal = get_goal_reach if environment_details["name"] == "fetch_Reach" else get_goal_push

policy_def = {'type': 'continuous', 'var': 1.0, 'max_action': 1.0, 'bias_action': 0.0}
baseline_hyper = {'n_critics': args.n_critics, 'n_actions': args.n_actions, 'conservative_weight': args.conservative_weight, 'expectile': args.expectile}


exp_name = environment_details["name"] + "_R=-" + str(exploration_strategy)
exp_name += "_" + algo_name + "_seed=" + str(seed)

writer = SummaryWriter(exp_folder+"logs/" + exp_name)

if not os.path.isdir(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(0)):
    os.mkdir(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(0))

''' Dataset and simulator '''
env = gym.make(environment_details["sim_name"], max_episode_steps=200)

file_name = exp_folder+"datasets/" + environment_details["name"] + "/" + "R=-" + str(exploration_strategy) + ".npz"
filenpz = np.load(file_name)
states, goals, actions, rewards, terminations, next_states = filenpz['arr_0'], filenpz['arr_1'], filenpz['arr_2'], filenpz['arr_3'], filenpz['arr_4'], filenpz['arr_5']

if algo_id in [0, 9]:
    offline_dataset = (states, goals, actions, rewards, terminations, next_states)
    dataset = Dataset_Custom(offline_dataset, device, batches_per_epoch * batch_size, get_goal, images=False, gamma=-1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
elif algo_id in [8]:
    offline_dataset = (states, goals, actions, rewards, terminations, next_states)
    dataset = Dataset_Custom(offline_dataset, device, batches_per_epoch * batch_size, get_goal, images=False, gamma=gamma)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
else:
    states = np.concatenate((states, goals), -1)
    dataset = d3rlpy.dataset.MDPDataset(states, actions, rewards, terminations)
    dataloader = None


''' Algorithm '''
sparse_reward_records, dense_norm_reward_records = [], []
logs_idx = 0
if algo_name == 'MetricRL':
    algo_class = None
    agent = MetricRL(input_dim,
                     goal_dim,
                     z_dim,
                     a_dim,
                     reg=reg,
                     policy_def=policy_def,
                     get_goal=get_goal,
                     device=device).to(device)

    for epoch in tqdm(range(EPOCHS)):
        avg_sparse_reward, avg_dense_score, logs_idx = train_loop(agent, dataloader, env, writer, logs_idx, epoch, reward_norm=reward_norm)
        sparse_reward_records.append(avg_sparse_reward)
        dense_norm_reward_records.append(avg_dense_score)
        np.savez(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(0) + "/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))


elif algo_name == 'QuasiMetric':
    algo_class = None
    agent = QuasiMetric(input_dim,
                        goal_dim,
                        z_dim,
                        a_dim,
                        get_goal=get_goal,
                        device=device).to(device)

    for epoch in tqdm(range(EPOCHS)):
        avg_sparse_reward, avg_dense_score, logs_idx = train_loop(agent, dataloader, env, writer, logs_idx, epoch, reward_norm=reward_norm)
        sparse_reward_records.append(avg_sparse_reward)
        dense_norm_reward_records.append(avg_dense_score)
        np.savez(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(0) + "/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))

elif algo_name == 'ContrastiveRL':
    algo_class = None
    offline_reg = 0.05
    agent = ContrastiveRL(input_dim,
                          goal_dim,
                          z_dim,
                          a_dim,
                          offline_reg,
                          policy_def=policy_def,
                          get_goal=get_goal,
                          device=device).to(device)

    for epoch in tqdm(range(EPOCHS)):
        avg_sparse_reward, avg_dense_score, logs_idx = train_loop(agent, dataloader, env, writer, logs_idx, epoch, reward_norm=reward_norm)
        sparse_reward_records.append(avg_sparse_reward)
        dense_norm_reward_records.append(avg_dense_score)
        np.savez(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(0) + "/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))

else:
    algo_class, agent = get_d3rlpy_algo(algo_name, gamma, batch_size, device_flag, baseline_hyper)

    def callback(algo: algo_class, epoch: int, total_step: int) -> None:
        avg_sparse_reward, avg_dense_score = simulation(agent, env, n_episodes=100, reward_norm=reward_norm)
        writer.add_scalar("Rewards/sparse_reward", avg_sparse_reward, epoch-1)
        writer.add_scalar("Rewards/dense_score", avg_dense_score, epoch-1)
        sparse_reward_records.append(avg_sparse_reward)
        dense_norm_reward_records.append(avg_dense_score)
        np.savez(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(0) + "/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))

    agent.fit(dataset,
              n_steps=batches_per_epoch*EPOCHS,
              n_steps_per_epoch=batches_per_epoch,
              experiment_name=None,
              save_interval=batches_per_epoch*EPOCHS+1,
              epoch_callback=callback,
              )


writer.close()









