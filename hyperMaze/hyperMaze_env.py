import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from my_utils.model_utils import fill_up_transitions, fill_up_actions


class HyperCube:

    def __init__(self, dim=2, size=100, max_T=1, use_obstacles=False, random_obstacles=True):

        self.dim = dim
        self.size = size
        self.step_size = 1./size
        self.max_T = max_T
        self.current_state = np.zeros(dim)
        self.goal = np.ones(dim)-self.step_size
        self.t = 0

        self.obstacles = None
        if use_obstacles:
            if random_obstacles:
                self.obstacles = np.reshape(np.random.randint(0, size, size=int(dim*(size**dim)*0.2)), (-1, dim)) / size
                a = np.nonzero(np.sum(((np.ones(dim)-self.step_size) - self.obstacles) ** 2, -1) == 0)[0]
                self.obstacles = np.delete(self.obstacles, a, axis=0)
                b = np.nonzero(np.sum((np.zeros(dim) - self.obstacles) ** 2, -1) == 0)[0]
                self.obstacles = np.delete(self.obstacles, b, axis=0)
            else:
                obst1_x = np.arange(int(0.2 * size), int(0.4 * size)) / size
                obst1_y = np.arange(int(0.0 * size), int(0.8 * size)) / size
                coord1 = [obst1_x] + [obst1_y] * (dim-1)
                all_meshes1 = np.meshgrid(*coord1)
                obst1 = np.concatenate([x.reshape(-1, 1) for x in all_meshes1], -1)

                obst2_x = np.arange(int(0.6 * size), int(0.8 * size)) / size
                obst2_y = np.arange(int(0.2 * size), int(1.0 * size)) / size
                coord2 = [obst2_x] + [obst2_y] * (dim-1)
                all_meshes2 = np.meshgrid(*coord2)
                obst2 = np.concatenate([x.reshape(-1, 1) for x in all_meshes2], -1)

                self.obstacles = np.concatenate((obst1, obst2), 0)

        all_actions, _ = fill_up_actions(np.zeros((3 ** self.dim, self.dim)), np.zeros(self.dim), self.dim, 0, 0)
        self.all_actions = all_actions.astype(float)
        print(end="")

    def reset(self, random_reset=False):
        self.t = 0
        if random_reset:
            resetted = False
            while not resetted:
                self.current_state = np.reshape(np.random.randint(0, self.size, size=self.dim), (self.dim)) / self.size
                if self.obstacles is None:
                    resetted = True
                else:
                    if np.nonzero(np.sum((self.current_state - self.obstacles) ** 2, -1) < 1e-6)[0].shape[0] == 0:
                        resetted = True
        else:
            self.current_state = np.zeros(self.dim)
        obs = {'observation': self.current_state, 'desired_goal': self.goal}#, 'obstacles': self.get_surrounding(self.current_state), 'desired_goal': self.goal}
        return obs, None

    def step(self, action, state=None):
        if state is not None:
            self.current_state = state

        next_state = self.current_state + action * self.step_size

        if (not np.any(next_state < 0)) and (not np.any(next_state > 1-1e-6)):
            if self.obstacles is not None:
                if np.nonzero(np.sum((next_state-self.obstacles)**2, -1) < 1e-6)[0].shape[0] == 0:
                    self.current_state = next_state
            else:
                self.current_state = next_state
        next_obs = {'observation': self.current_state, 'desired_goal': self.goal}#, 'obstacles': self.get_surrounding(self.current_state), 'desired_goal': self.goal}
        reward = 0.0
        term = False
        if np.linalg.norm(self.current_state-self.goal) < self.step_size:
            reward = 1.0
            term = True
        # if self.t >= self.max_T:
        #     term = True
        # self.t += 1
        return next_obs, reward, term, False, None

    def get_random_actions(self):
        a_idx = np.random.randint(0, self.all_actions.shape[0])
        a = self.all_actions[a_idx]
        return a, a_idx

    def check_if_obst(self, state):
        if self.obstacles is not None:
            if np.nonzero(np.sum((state - self.obstacles) ** 2, -1) < 1e-6)[0].shape[0] > 0:
                return True
        return False

    # def get_surrounding(self, pos):
    #
    #     neigh_obst = np.zeros(self.all_actions.shape[0])
    #     for i, a in enumerate(self.all_actions):
    #         next_state = pos + a * self.step_size
    #         if (np.any(next_state < 0)) or (np.any(next_state > 1 - 1e-6)):
    #             neigh_obst[i] = 1
    #         if np.nonzero(np.sum((next_state - self.obstacles) ** 2, -1) < 1e-6)[0].shape[0] > 0:
    #             neigh_obst[i] = 1
    #
    #     return neigh_obst























