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

from miniGrid_envs import KeyEnv, OpenEnv
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper


def img_preprocess(img):
    return np.ascontiguousarray(np.transpose(img, (0, 3, 1, 2))) / 255.


def img_preprocess_d3rlpy(img):
    return np.ascontiguousarray(np.transpose(img, (0, 3, 1, 2))).astype(np.uint8)


def get_goal_open(goal):
    return goal


def get_goal_key(goal):
    return np.concatenate((goal, np.ones((goal.shape[0], 3))*np.array([[-1, -1, 0]])), -1)

def set_action_discrete_argmax(a):
    return np.argmax(a)


def set_action_discrete(a):
    return a[0]


def set_action_continuous(a):
    return int(np.clip(np.round(np.squeeze(a)), 0, norm_a-1))



if os.getcwd().split('/')[-1] == 'miniGrid':
    exp_folder = './'
    root_folder = '../'
elif os.getcwd().split('/')[-1] == 'cluster_sh':
    exp_folder = '../miniGrid/'
    root_folder = '../'
else:
    exp_folder = './miniGrid/'
    root_folder = './'

with open(root_folder+'config.yml', 'r') as file:
    configs = yaml.safe_load(file)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
''' Dataset and Algorithm '''
parser.add_argument('--environment', default=0, type=int)
parser.add_argument('--algorithm', default=8, type=int)
parser.add_argument('--use_images', default=0, type=int)
parser.add_argument('--epsilon', default=0.1, type=float)
''' MetricRL HyperParameters '''
parser.add_argument('--z_dim', default=128, type=int)
parser.add_argument('--reg', default=1.0, type=float, help="contrastive negative")
''' Baselines HyperParameters '''
parser.add_argument('--n_critics', default=2, type=int)
parser.add_argument('--n_actions', default=10, type=int)
parser.add_argument('--conservative_weight', default=5.0, type=float)
parser.add_argument('--expectile', default=0.99, type=float)
args = parser.parse_args()


seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_flag = False
if args.use_images == 1:
    device_flag = torch.cuda.is_available()

''' HyperParameters '''
reward_norm = configs['miniGrid']["reward_norm"]
epsilon = args.epsilon
algo_id = args.algorithm
algo_name = configs["algorithms"][algo_id]
environment_details = configs['miniGrid']["experiments"][args.environment]
EPOCHS, batches_per_epoch, batch_size, gamma = configs["EPOCHS"], configs["batches_per_epoch"], configs["batch_size"], configs["gamma"]
input_dim, goal_dim, a_dim = environment_details["state_dim"], environment_details["goal_dim"], environment_details["action_dim"]


z_dim = args.z_dim
reg = args.reg
use_images = args.use_images == 1

policy_def = {'type': 'discrete', 'var': 1.0, 'max_action': 1.0, 'bias_action': 0.0}
baseline_hyper = {'n_critics': args.n_critics, 'n_actions': args.n_actions, 'conservative_weight': args.conservative_weight, 'expectile': args.expectile}



exp_name = environment_details["name"] + "_eps=" + str(epsilon)
exp_name += "_" + algo_name + "_use_images=" + str(use_images)
exp_name += "_seed=" + str(seed)

writer = SummaryWriter(exp_folder+"logs/" + exp_name + "")

if not os.path.isdir(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(use_images)):
    os.mkdir(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(use_images))

''' Dataset and simulator '''

file_name = exp_folder+"datasets/"+environment_details['name']+"_img="+str(use_images)+"_eps="+str(epsilon)+".npz"
filenpz = np.load(file_name)
states, goals, actions, rewards, terminations, next_states = filenpz['arr_0'], filenpz['arr_1'], filenpz['arr_2'], filenpz['arr_3'], filenpz['arr_4'], filenpz['arr_5']

T = 100

state_preprocess = lambda x: x
get_goal = get_goal_open
if use_images:
    if args.environment == 0:
        env = OpenEnv(use_images=True, max_steps=T, size=10)  # render_mode="human"
        env = RGBImgObsWrapper(env)  # get directly obs['image']
    else:
        env = KeyEnv(use_images=True, max_steps=T)  # render_mode="human"
        env = RGBImgObsWrapper(env)  # get directly obs['image']

    if algo_id in [0, 8, 9]:
        state_preprocess = img_preprocess
        states = state_preprocess(states)
        goals = state_preprocess(goals)
        next_states = state_preprocess(next_states)
    else:
        state_preprocess = img_preprocess_d3rlpy
        states = state_preprocess(states)
else:
    if args.environment == 0:
        env = OpenEnv(use_images=False, max_steps=T, size=10)  # render_mode="human"
        env = FullyObsWrapper(env)  # get directly obs['image']
    else:
        env = KeyEnv(use_images=False, max_steps=T)  # render_mode="human"
        env = FullyObsWrapper(env)  # get directly obs['image']
        get_goal = get_goal_key

    if algo_id == 0 or algo_id == 8 or algo_id == 9:
        if args.environment == 0:
            states, goals, next_states = states[:, :2], goals[:, :2], next_states[:, :2]
        else:
            states, goals, next_states = states[:, :5], goals[:, :5], next_states[:, :5]




set_action = set_action_discrete_argmax
if algo_id in [0, 9]:
    offline_dataset = (states, goals, actions, rewards, terminations, next_states)
    dataset = Dataset_Custom(offline_dataset, device, batches_per_epoch * batch_size, get_goal_open, images=use_images, gamma=-1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
elif algo_id in [8]:
    offline_dataset = (states, goals, actions, rewards, terminations, next_states)
    dataset = Dataset_Custom(offline_dataset, device, batches_per_epoch * batch_size, get_goal_open, images=use_images, gamma=gamma)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
else:
    set_action = set_action_discrete

    norm_a = None
    if algo_id in [1, 5, 6, 7]:
        norm_a = a_dim
        actions = (actions.astype(float) + np.random.randn(actions.shape[0], actions.shape[1])*0.05)
        set_action = set_action_continuous

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
                     use_images=use_images,
                     device=device).to(device)

    for epoch in tqdm(range(EPOCHS)):
        avg_sparse_reward, avg_dense_score, logs_idx = train_loop(agent, dataloader, env, writer, logs_idx, epoch, set_action=set_action, use_images=use_images, state_preprocess=state_preprocess, reward_norm=reward_norm)
        sparse_reward_records.append(avg_sparse_reward)
        dense_norm_reward_records.append(avg_dense_score)
        np.savez(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(use_images) + "/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))

elif algo_name == 'QuasiMetric':
    algo_class = None
    agent = QuasiMetric(input_dim,
                        goal_dim,
                        z_dim,
                        a_dim,
                        get_goal=get_goal,
                        use_images=use_images,
                        device=device).to(device)

    for epoch in tqdm(range(EPOCHS)):
        avg_sparse_reward, avg_dense_score, logs_idx = train_loop(agent, dataloader, env, writer, logs_idx, epoch, set_action=set_action, use_images=use_images, state_preprocess=state_preprocess, reward_norm=reward_norm)
        sparse_reward_records.append(avg_sparse_reward)
        dense_norm_reward_records.append(avg_dense_score)
        np.savez(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(use_images) + "/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))

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
                          use_images=use_images,
                          device=device).to(device)

    for epoch in tqdm(range(EPOCHS)):
        avg_sparse_reward, avg_dense_score, logs_idx = train_loop(agent, dataloader, env, writer, logs_idx, epoch, set_action=set_action, use_images=use_images, state_preprocess=state_preprocess, reward_norm=reward_norm)
        sparse_reward_records.append(avg_sparse_reward)
        dense_norm_reward_records.append(avg_dense_score)
        np.savez(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(use_images) + "/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))

else:
    algo_class, agent = get_d3rlpy_algo(algo_name, gamma, batch_size, device_flag, baseline_hyper, use_images=use_images, discrete=True)

    def callback(algo: algo_class, epoch: int, total_step: int) -> None:
        avg_sparse_reward, avg_dense_score = simulation(agent, env, n_episodes=100, set_action=set_action, use_images=use_images, state_preprocess=state_preprocess, reward_norm=reward_norm)
        # avg_sparse_reward = simulation(env, agent, T, input_dim, use_images=use_images, norm_a=norm_a)
        writer.add_scalar("Rewards/sparse_reward", avg_sparse_reward, epoch-1)
        sparse_reward_records.append(avg_sparse_reward)
        dense_norm_reward_records.append(avg_dense_score)
        np.savez(root_folder+"saved_results/" + environment_details["name"] + "_img=" + str(use_images) + "/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))


    agent.fit(dataset,
              n_steps=batches_per_epoch*EPOCHS,
              n_steps_per_epoch=batches_per_epoch,
              experiment_name=None,
              save_interval=batches_per_epoch*EPOCHS+1,
              epoch_callback=callback,
              show_progress=False,
              )


writer.close()



