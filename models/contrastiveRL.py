import numpy as np
import torch
import torch.nn as nn
from my_utils.model_utils import MLP, CNN, Policy, Memory, fill_up_actions
import time
from torch.distributions.uniform import Uniform


class ContrastiveRL(nn.Module):

    def __init__(self, input_dim, goal_dim, z_dim, a_dim, offline_reg, device=None, use_images=0, policy_def=None, get_goal=None):
        super().__init__()

        self.lambd = offline_reg
        self.device = device
        self.input_dim = input_dim

        self.use_images = use_images
        self.get_goal = get_goal
        self.discrete = True if policy_def['type'] == 'discrete' else False

        a_dim_cond = 1 if self.discrete else a_dim

        actor_params = []
        if use_images == 0:
            self.phi = MLP(input_dim, a_dim_cond, z_dim, 3, residual=True)
            self.psi = MLP(input_dim, 0, z_dim, 3, residual=True)
            policy_input_dim, policy_cond_dim = input_dim, input_dim #, goal_dim
            self.pi_encoder = lambda x: x
        else:
            self.phi = CNN(3, a_dim_cond, z_dim)
            self.psi = CNN(3, 0, z_dim)
            policy_input_dim, policy_cond_dim = z_dim, 0
            self.pi_encoder = CNN(3, 0, z_dim)
            actor_params += list(self.pi_encoder.parameters())

        self.pi = Policy(policy_input_dim, policy_cond_dim, a_dim, policy_def)
        actor_params += list(self.pi.parameters())

        self.opt_critic = torch.optim.Adam(list(self.phi.parameters()) + list(self.psi.parameters()), lr=1e-3)
        self.opt_actor = torch.optim.Adam(actor_params, lr=1e-3)


    def predict(self, s):

        if self.use_images == 0:
            st = torch.from_numpy(s[:, :self.input_dim]).float().to(self.device)
            gt = torch.from_numpy(self.get_goal(s[:, self.input_dim:])).float().to(self.device)
        else:
            st = self.pi_encoder(torch.from_numpy(s).float().to(self.device))
            gt = None

        mu = self.pi.get_mean(st, gt)
        return torch.squeeze(mu).detach().cpu().numpy()

    def critic_loss(self, st, at, st1):

        z_sa = self.phi(st, at)
        z_g = self.psi(st1)

        logits = torch.einsum('ik, jk->ij', z_sa, z_g)
        L_pos = nn.functional.binary_cross_entropy_with_logits(logits, torch.eye(logits.shape[0]).to(self.device))

        return L_pos

    def actor_loss(self, st, at, goal_t):

        state = self.pi_encoder(st)
        goal_state = None if self.use_images == 1 else goal_t

        a = self.pi.sample_action(state, goal_state)
        if self.discrete:
            a = a.view(-1, 1)
        z_sa = self.phi(st, a)
        z_g = self.psi(goal_t)
        logits = torch.einsum('ik, ik->i', z_sa, z_g)

        log_prob_a_orig, _ = self.pi.get_log_prob(state, at, goal_state)

        tot_loss = (1 - self.lambd) * torch.mean(-1.0 * logits) - self.lambd * torch.mean(log_prob_a_orig)

        return tot_loss

    def update(self, batch=None, batch_size=256, epoch=0, train_modules=None):

        st, gt, at, _, _, st1 = batch

        st = st.float() if st.is_cuda else st.float().to(self.device)
        gt = gt.float() if gt.is_cuda else gt.float().to(self.device)
        at = at.float() if at.is_cuda else at.float().to(self.device)
        st1 = st1.float() if st1.is_cuda else st1.float().to(self.device)

        L_pos = self.critic_loss(st, at, st1)
        critic_loss = L_pos

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        L_pi = self.actor_loss(st, at, gt)

        self.opt_actor.zero_grad()
        L_pi.backward()
        self.opt_actor.step()

        logs_losses = {'L_pos': L_pos.detach().cpu().item() if torch.is_tensor(L_pos) else 0,
                       'L_neg': 0,
                       'L_trans': 0,
                       'L_pi': L_pi.detach().cpu().item() if torch.is_tensor(L_pi) else 0,
                       'L_DQN': 0}

        return logs_losses

    def update_target(self, episode):
        return

