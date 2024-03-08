import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, CenterCrop, Grayscale


def simulation(model, sim, n_episodes=10, use_images=0, set_action=lambda x: x, state_preprocess=lambda x: x, reward_norm=0):
    total_sparse_reward, total_dense_reward = 0, 0
    for i in range(n_episodes):
        sparse_reward, dense_reward = 0, 0

        obs, goal = sim.reset()

        if use_images == 1:
            augmented_state = state_preprocess(np.expand_dims(obs['image'], 0))
        else:
            if "desired_goal" in obs:
                augmented_state = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
            else:
                augmented_state = obs['observation']

        for t in range(1000):
            current_action = set_action(model.predict(augmented_state))
            next_obs, reward, terminated, truncated, info = sim.step(np.squeeze(current_action))

            sparse_reward += reward + reward_norm

            if "achieved_goal" in next_obs:
                distance = np.linalg.norm(next_obs['achieved_goal'] - next_obs['desired_goal'], axis=-1)
                dense_reward += np.exp(-distance)

            obs = next_obs
            if use_images == 1:
                augmented_state = state_preprocess(np.expand_dims(obs['image'], 0))
            else:
                if "desired_goal" in obs:
                    augmented_state = np.expand_dims(np.concatenate((obs['observation'], obs['desired_goal']), -1), 0)
                else:
                    augmented_state = obs['observation']

            if terminated or truncated:# or solved > 0:
                break

        total_sparse_reward += sparse_reward
        total_dense_reward += dense_reward

    return total_sparse_reward / n_episodes, total_dense_reward / n_episodes


def train_loop(agent, dataloader, env, writer, logs_idx, epoch, use_images=0, set_action=lambda x: x, state_preprocess=lambda x: x, reward_norm=0):
    # sparse_reward_records, dense_norm_reward_records = [], []
    # logs_idx = 0
    # for epoch in tqdm(range(EPOCHS)):

    agent.train()
    for loader_idx, batch in enumerate(dataloader):

        if epoch < 10:
            train_modules = (1, 0)
        else:
            train_modules = (1, 1)
        log_losses = agent.update(batch=batch, epoch=epoch, train_modules=train_modules)

        writer.add_scalar("Losses/positive_loss", log_losses['L_pos'], logs_idx)
        writer.add_scalar("Losses/negative_loss", log_losses['L_neg'], logs_idx)
        writer.add_scalar("Losses/actor_loss", log_losses['L_pi'], logs_idx)
        if log_losses['L_trans'] is not None:
            writer.add_scalar("Losses/transition_loss", log_losses['L_trans'], logs_idx)
        logs_idx += 1

    #if epoch % 1 == 0:
    agent.eval()
    avg_sparse_reward, avg_dense_score = simulation(agent, env, n_episodes=100, use_images=use_images, set_action=set_action, state_preprocess=state_preprocess, reward_norm=reward_norm)

    writer.add_scalar("Rewards/sparse_reward", avg_sparse_reward, epoch)
    writer.add_scalar("Rewards/dense_score", avg_dense_score, epoch)
    return avg_sparse_reward, avg_dense_score, logs_idx
    #
    # sparse_reward_records.append(avg_sparse_reward)
    # dense_norm_reward_records.append(avg_dense_score)
    # np.savez("../saved_results/" + exp_env + "_img=" + str(use_images) + "/" + exp_name + ".npz", np.array(sparse_reward_records), np.array(dense_norm_reward_records))


















