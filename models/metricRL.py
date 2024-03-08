import numpy as np
import torch
import torch.nn as nn
from my_utils.model_utils import MLP, CNN, Policy, Memory, fill_up_actions
import time
from torch.distributions.uniform import Uniform


'''
    transition: None
    policy_gradient continous: var = 1.0, max_action = 1.0, bias_action = 0.0
    policy gradient discrete: None
    max_action=1, bias_action=0, discrete=False, all_actions_idx=None, all_actions=None
'''

class MetricRL(nn.Module):

    def __init__(self, input_dim, goal_dim, z_dim, a_dim, memory_size=0, reg=1.0, gamma=1.0, device=None, use_images=0, policy_def=None, get_goal=None):
        super().__init__()

        self.replay_buffer = Memory(memory_size)
        # self.all_actions_idx = all_actions_idx
        # self.all_actions = all_actions

        self.reg = reg

        self.device = device

        self.input_dim = input_dim
        self.goal_dim = goal_dim
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.gamma = gamma

        self.get_goal = get_goal

        self.use_images = use_images

        actor_params = []
        if use_images == 0:
            self.phi = MLP(input_dim, 0, z_dim, 3, residual=True)
            self.pi_encoder = lambda x: x
            policy_input_dim, policy_cond_dim = input_dim, input_dim #, goal_dim
        else:
            self.phi = CNN(3, 0, z_dim)
            self.pi_encoder = CNN(3, 0, z_dim)
            policy_input_dim, policy_cond_dim = z_dim, 0
            actor_params += list(self.pi_encoder.parameters())
        critic_params = list(self.phi.parameters())

        if policy_def['type'] == 'transition':
            self.T = MLP(input_dim, a_dim, input_dim, 2)
            self.pi = None
            critic_params += list(self.T.parameters())
        else:
            self.T = None
            self.pi = Policy(policy_input_dim, policy_cond_dim, a_dim, policy_def)
            actor_params += list(self.pi.parameters())

        self.opt_critic = torch.optim.Adam(critic_params, lr=1e-3)
        self.opt_actor = None if len(actor_params) == 0 else torch.optim.Adam(actor_params, lr=1e-3)


    def get_distance(self, z1, z2):
        dist = torch.squeeze(torch.cdist(torch.unsqueeze(z1, 1), z2, p=2.0))
        return dist

    def get_multi_distance(self, z1, z2):
        dist = torch.linalg.vector_norm(z1 - z2, ord=2, dim=-1)
        return dist

    def forward_dynamics(self, s, a):
        return s + self.T(s, a)

    def predict(self, s):

        if self.use_images == 0:
            st = torch.from_numpy(s[:, :self.input_dim]).float().to(self.device)
            gt = torch.from_numpy(self.get_goal(s[:, self.input_dim:])).float().to(self.device)
        else:
            st = self.pi_encoder(torch.from_numpy(s).float().to(self.device))
            gt = None

        mu = self.pi.get_mean(st, gt)
        return torch.squeeze(mu).detach().cpu().numpy()

    def get_action(self, state, current_obst, goal, test=False):

        if not test and np.random.rand() < self.epsilon:
            a_idx = np.random.randint(0, self.all_actions.shape[0])
        else:
            zg = self.phi(goal)
            all_next_s = state.repeat(self.dim_a, 1) + self.T(state.repeat(self.dim_a, 1), torch.cat([torch.from_numpy(self.all_actions).view(-1, self.dim_input).float().to(self.device), current_obst.repeat(self.dim_a, 1)], -1))
            next_states_dist = self.get_distance(self.phi(all_next_s), zg)
            a_idx = torch.argmin(next_states_dist).detach().cpu().item()
        return a_idx

    def get_value(self, s, s_g):

        z = self.phi(s)
        z_g = self.phi(s_g)
        neg_dist = - self.get_distance(z, z_g)
        return neg_dist

    def get_multi_value(self, s, goals, goal_rewards):

        z = self.phi(s)
        z_g = self.phi(goals)

        dZ = self.get_distance(z, z_g)
        V = torch.max((self.gamma ** dZ) * torch.squeeze(goal_rewards), -1)[0]

        return V

    def critic_loss(self, st, at, st1):

        z = self.phi(st)
        z1 = self.phi(st1)

        action_distance = 1
        L_pos = torch.mean((self.get_distance(z1, z) - action_distance) ** 2)

        idx = np.zeros((z.shape[0], z.shape[0] - 1))
        for i in range(z.shape[0]):
            idx[i] = np.delete(np.arange(z.shape[0]), i)
        z1_rep = torch.cat([torch.unsqueeze(z1[i], 0) for i in idx], 0)
        dist_z_perm = - torch.log(torch.cdist(torch.unsqueeze(z, 1), z1_rep) + 1e-6)
        L_neg = torch.mean(dist_z_perm)

        return L_pos, L_neg

    def actor_loss(self, st, s_g, st1, at):

        state = self.pi_encoder(st)
        goal_state = None if self.use_images == 1 else s_g

        # DDPG like
        # at = self.pi(st, goal_state)
        # s1_pred = self.T(st, at)
        # V_next = self.get_value(s1_pred, s_g)
        # tot_loss = - torch.mean(V_next)

        log_prob, entropy = self.pi.get_log_prob(state, at, goal_state)
        Vt = self.get_value(st, s_g)
        Vt1 = self.get_value(st1, s_g)
        Adv = (Vt1 - Vt).detach()
        tot_loss = - torch.mean(Adv * log_prob)

        return tot_loss

    def update(self, batch=None, batch_size=256, epoch=0, train_modules=(1, 1)):

        if batch is None:
            data = self.replay_buffer.sample(batch_size)

            st = torch.cat([x['st'] for x in data], 0)
            obst = torch.cat([x['obst'] for x in data], 0)
            gt = torch.cat([x['goal_t'] for x in data], 0)
            at = torch.cat([torch.from_numpy(x['a']).float().view(1, -1).to(st.device) for x in data], 0)
            a_idx = np.array([x['a_idx'] for x in data])
            st1 = torch.cat([x['st1'] for x in data], 0)
        else:
            st, gt, at, _, _, st1 = batch #, ot, og, ot1 = batch

            st = st.float() if st.is_cuda else st.float().to(self.device)
            gt = gt.float() if gt.is_cuda else gt.float().to(self.device)
            at = at.float() if at.is_cuda else at.float().to(self.device)
            st1 = st1.float() if st1.is_cuda else st1.float().to(self.device)

        L_pos, L_neg, L_trans, L_pi = 0, 0, 0, 0

        if train_modules[0] == 1:

            L_pos, L_neg = self.critic_loss(st, at, st1)

            if self.T is not None:
                s1_hat = self.forward_dynamics(st, at)
                L_trans = torch.mean(torch.sum((st1 - s1_hat) ** 2, -1))

            critic_loss = L_pos + self.reg * L_neg + L_trans

            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()

        if train_modules[1] == 1:

            L_pi = self.actor_loss(st, gt, st1, at)  # , ot=ot, og=og, ot1=ot1)

            self.opt_actor.zero_grad()
            L_pi.backward()
            # if self.policy_clip is not None:
            #     torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.policy_clip)
            self.opt_actor.step()

        logs_losses = {'L_pos': L_pos.detach().cpu().item() if torch.is_tensor(L_pos) else L_pos,
                       'L_neg': L_neg.detach().cpu().item() if torch.is_tensor(L_neg) else L_neg,
                       'L_trans': L_trans.detach().cpu().item() if torch.is_tensor(L_trans) else L_trans,
                       'L_pi': L_pi.detach().cpu().item() if torch.is_tensor(L_pi) else L_pi,
                       'L_DQN': 0}

        return logs_losses

    def update_target(self, episode):
        return

