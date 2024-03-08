import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.distributions.geometric import Geometric


'''
    rewards is always zero, last step before termination is discarded
'''
class Dataset_Custom(Dataset):

    def __init__(self, data, device, size, get_goal, images=False, gamma=-1):

        states, goals, actions, rewards, terminations, next_states = data[0], data[1], data[2], data[3], data[4], data[5]

        self.device = device
        self.images = images
        self.GEOM = None if gamma <= 0 else Geometric(1 - gamma)

        trj_idx = np.nonzero(terminations[:, 0])[0]

        self.states, self.goals, self.actions, self.rewards, self.terminations, self.next_states = [], [], [], [], [], []
        i = 0
        for j in trj_idx:
            if not self.images:
                self.states.append(torch.from_numpy(states[i:j+1]).float().to(self.device))
                self.goals.append(torch.from_numpy(get_goal(goals[i:j+1])).float().to(self.device))
                # self.next_states.append(torch.from_numpy(states[i+1:j+1]).float().to(self.device))
                self.next_states.append(torch.from_numpy(next_states[i:j+1]).float().to(self.device))
                self.actions.append(torch.from_numpy(actions[i:j+1]).float().to(self.device))
                self.rewards.append(torch.from_numpy(rewards[i:j+1]).float().to(self.device))
                self.terminations.append(torch.from_numpy(terminations[i:j+1]).float().to(self.device))
            else:
                self.states.append(torch.cat([torch.from_numpy(states[i:j+1]).float(),
                                              torch.from_numpy(next_states[j:j+1]).float()], 0))
                self.goals.append(torch.cat([torch.from_numpy(goals[i:j+1]).float(),
                                             torch.from_numpy(goals[j:j+1]).float()], 0))
                self.next_states.append(torch.cat([torch.from_numpy(next_states[i:j+1]).float(),
                                                   torch.from_numpy(next_states[j:j+1] * 0).float()], 0))
                self.actions.append(torch.cat([torch.from_numpy(actions[i:j+1]).float().to(self.device),
                                               torch.from_numpy(np.array([[0]])).float().to(self.device)], 0))
                self.rewards.append(torch.cat([torch.from_numpy(rewards[i:j+1]).float().to(self.device),
                                               torch.from_numpy(np.array([[0]])).float().to(self.device)], 0))
                self.terminations.append(torch.cat([torch.from_numpy(terminations[i:j+1]).float().to(self.device),
                                                    torch.from_numpy(np.array([[1]])).float().to(self.device)], 0))

            i = j + 1

        self.n_trj = len(self.states)
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        found_trj = False
        while not found_trj:
            trj_i = np.random.randint(0, self.n_trj)
            len_trj = self.states[trj_i].shape[0]
            if len_trj > 0:
                found_trj = True
        t = 0 if self.GEOM is None else int(torch.clamp(self.GEOM.sample(), max=len_trj - 1).item())
        i = np.random.randint(0, len_trj - t)

        s = self.states[trj_i][i]
        g = self.goals[trj_i][i]
        a = self.actions[trj_i][i]
        r = self.rewards[trj_i][i]
        term = self.terminations[trj_i][i]
        s1 = self.next_states[trj_i][i+t]

        return s, g, a, r, term, s1


class Dataset_Minari(Dataset):
    def __init__(self, data, device, size, get_goal, gamma=-1):

        self.device = device
        self.GEOM = None if gamma <= 0 else Geometric(1 - gamma)

        self.states, self.goals, self.actions, self.rewards, self.terminations, self.next_states = [], [], [], [], [], []
        for e in tqdm(data.episode_indices):
            trj = data[e]
            if trj.observations['observation'].shape[0] < 2 or trj.actions.shape[0] < 1:
                print("empty trj", e)
            elif (trj.observations['observation'].shape[0] - 1) == trj.actions.shape[0]:
                self.states.append(torch.from_numpy(trj.observations['observation'][:-1]).float().to(self.device))
                self.goals.append(torch.from_numpy(get_goal(trj.observations['desired_goal'][:-1])).float().to(self.device))
                self.next_states.append(torch.from_numpy(trj.observations['observation'][1:]).float().to(self.device))
                self.actions.append(torch.from_numpy(trj.actions).float().to(self.device))
                self.rewards.append(torch.from_numpy(trj.rewards).float().to(self.device))
                self.terminations.append(torch.from_numpy(trj.terminations*1.).float().to(self.device))
            else:
                print("non complete trj", e)

        self.n_trj = len(self.states)
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        trj_i = np.random.randint(0, self.n_trj)
        len_trj = self.states[trj_i].shape[0]
        t = 0 if self.GEOM is None else int(torch.clamp(self.GEOM.sample(), max=len_trj - 1).item())
        i = np.random.randint(0, len_trj - t)

        s = self.states[trj_i][i]
        g = self.goals[trj_i][i]
        a = self.actions[trj_i][i]
        r = self.rewards[trj_i][i]
        term = self.terminations[trj_i][i]
        s1 = self.next_states[trj_i][i+t]

        return s, g, a, r, term, s1


