import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from my_utils.model_utils import MLP, Memory, fill_up_actions


class DQN(nn.Module):

    def __init__(self, dim_input, dim_a, gamma, memory_size, epsilon, tau, curiosity_lambda=0, n_layers=4, device="cpu"):  # n = dimensionality output actions, D = memory buffer already filled
        super(DQN, self).__init__()

        self.replay_buffer = Memory(memory_size)
        self.device = device

        self.epsilon = epsilon#0.5#epsilon
        self.tau = tau #0.1 #0.005
        self.curiosity_lambda = curiosity_lambda

        self.update_target_episodes = 1#9#100
        self.gamma = gamma

        self.dim_input = dim_input
        self.dim_a = dim_a

        # all_actions, _ = fill_up_actions(np.zeros((3 ** self.dim_input, self.dim_input)), np.zeros(self.dim_input), self.dim_input, 0, 0)
        # self.all_actions = torch.from_numpy(all_actions).float().to(device)

        self.Q_model = MLP(dim_input, 0, dim_a, n_layers, residual=True)

        # Target network updated only once every self.update_target_episodes
        self.Q_target = MLP(dim_input, 0, dim_a, n_layers, residual=True)
        self.Q_target.load_state_dict(self.Q_model.state_dict())
        self.Q_target.eval()

        self.opt_critic = torch.optim.Adam(self.Q_model.parameters(), lr=1e-3)

        ############################ Curiosity Driven
        self.f_curious = MLP(dim_input, 1, dim_input, n_layers, residual=True)
        self.opt_curious = torch.optim.Adam(self.f_curious.parameters(), lr=1e-3)


        #self.opt_critic = torch.optim.AdamW(self.Q_model.parameters(), lr=1e-3, amsgrad=True)

    def get_value(self, s, s_g=None):
        q_value = self.Q_target(s)
        return torch.max(q_value).detach().cpu().item()

    def get_action(self, state, current_obst, goal, test=False):

        if not test and np.random.rand() < self.epsilon:
            #a_idx = np.random.randint(0, self.all_actions.shape[0])
            a_idx = np.random.randint(0, self.dim_a)
        else:
            q = self.Q_target(state)
            a_idx = torch.argmax(q).detach().cpu().item()

        return a_idx

    def update(self, batch_size=256):

        if len(self.replay_buffer) < 1000:
            return None

        self.Q_model.train()

        data = self.replay_buffer.sample(batch_size)

        st = torch.cat([x['st'] for x in data], 0)
        a_idx = np.array([x['a_idx'] for x in data])
        st1 = torch.cat([x['st1'] for x in data], 0)
        r = torch.cat([torch.tensor(x['r'], dtype=torch.float32).view(1) for x in data], 0).to(self.device)
        terminal = torch.cat([torch.tensor(x['terminal'] * 1.0, dtype=torch.float32).view(1) for x in data], 0).to(self.device)

        # CURIOSITY
        at = torch.from_numpy(np.expand_dims(a_idx, -1)).float().to(st.device)
        st1_hat = self.f_curious(st, at)
        err_pred = torch.mean((st1 - st1_hat)**2, -1)
        self.opt_curious.zero_grad()
        (torch.mean(err_pred)).backward()
        self.opt_curious.step()
        r += self.curiosity_lambda * err_pred

        # Compute value of st from target network by r + gamma* argmax(Q_target(st1))
        Qt1 = self.Q_target(st1)
        max_vals, _ = torch.max(Qt1, -1)
        y = (r + terminal * (self.gamma * max_vals)).detach()

        # Compute value of st from Q_value network by Q(st) and get the Q value just for the action given from the buffer
        Q = self.Q_model(st)
        Q = torch.cat([Q[i, idx].view(1) for i, idx in enumerate(a_idx)], 0)

        # Compute the loss that corresponds to the Temporal Difference error
        TDerror = (y - Q) ** 2
        loss = torch.mean(TDerror)

        self.opt_critic.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(self.Q_model.parameters(), 100)
        self.opt_critic.step()

        logs_losses = {'L_pos': 0,
                       'L_neg': 0,
                       'L_trans': 0,#torch.mean(err_pred),
                       'L_DQN': loss.detach().cpu().item()}

        return logs_losses, torch.sum(r).detach().cpu().item()

    def update_target(self, episode):

        if episode % self.update_target_episodes == self.update_target_episodes-1:
            target_net_state_dict = self.Q_target.state_dict()
            policy_net_state_dict = self.Q_model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.Q_target.load_state_dict(target_net_state_dict)
            self.Q_target.eval()

