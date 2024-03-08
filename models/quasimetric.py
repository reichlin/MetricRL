import numpy as np
import torch
import torch.nn as nn
from my_utils.model_utils import MLP, Policy
import time
from torch.distributions.uniform import Uniform


class QuasiMetric(nn.Module):

    def __init__(self, input_dim, goal_dim, z_dim, a_dim, device=None, use_images=0, get_goal=None):
        super().__init__()

        self.device = device
        self.get_goal = get_goal

        self.input_dim = input_dim
        self.goal_dim = goal_dim
        self.z_dim = z_dim
        self.a_dim = a_dim

        self.f = MLP(input_dim, 0, z_dim, 4)
        self.d = MLP(2*z_dim, 0, 1, 4)
        self.T = MLP(z_dim, a_dim, z_dim, 4)
        self.pi = MLP(z_dim, goal_dim, a_dim, 4)

        self.opt_critic = torch.optim.Adam(list(self.f.parameters())+list(self.d.parameters())+list(self.T.parameters()), lr=1e-3)
        self.opt_actor = torch.optim.Adam(self.pi.parameters(), lr=1e-3)


    def get_distance(self, x, y):
        D = x.shape[-1]
        valid = x < y
        xy = torch.cat(torch.broadcast_tensors(x, y), dim=-1)
        sxy, ixy = xy.sort(dim=-1)
        neg_inc_copies = torch.gather(valid, dim=-1, index=ixy % D) * torch.where(ixy < D, -1, 1)
        neg_inp_copies = torch.cumsum(neg_inc_copies, dim=-1)
        neg_f = (neg_inp_copies < 0) * (-1.)
        neg_incf = torch.cat([neg_f.narrow(-1, 0, 1), torch.diff(neg_f, dim=-1)], dim=-1)
        return (sxy * neg_incf).sum(-1)

    def predict(self, s):
        # TODO: finish me
        if self.use_images == 0:
            st = torch.from_numpy(s[:, :self.input_dim]).float().to(self.device)
            gt = torch.from_numpy(self.get_goal(s[:, self.input_dim:])).float().to(self.device)
        else:
            st = self.pi_encoder(torch.from_numpy(s).float().to(self.device))
            gt = None
        z = self.f(st)
        a = torch.tanh(self.pi(z, gt))
        return torch.squeeze(a).detach().cpu().numpy()

    def critic_loss(self, st, sg, at, rt, st1):

        zt = self.f(st)
        zg = self.f(sg)
        zt1 = self.f(st1)
        zt1_hat = self.T(zt, at)

        epsilon = 0.25
        lamb = 0.01
        softplus = nn.Softplus()
        relu = nn.ReLU()
        # dist_goal = torch.mean(-100 * softplus(5 - self.d(torch.cat([zt, zg], -1))))
        # dist_goal = torch.mean(-100 * softplus(5 - self.d(torch.cat([zt, zg], -1)) / 100))
        #dist_goal = torch.mean(- softplus(1 - self.d(torch.cat([zt, zg], -1))))

        idx = np.arange(zt.shape[0])
        np.random.shuffle(idx)
        zr = zt1[idx]
        dist_goal = torch.mean(-100 * softplus(5 - self.get_distance(zt, zr)/100))

        # dist_goal = - torch.mean(torch.log(self.d(torch.cat([zt, zg], -1))))
        # dist_states = torch.mean(relu(self.d(torch.cat([zt, zt1], -1)) - 1)**2) - epsilon**2
        dist_states = torch.mean(relu(self.get_distance(zt, zt1) - 1) ** 2) - epsilon ** 2

        L_qm = - dist_goal + lamb * dist_states

        # L_trans = torch.mean(self.get_distance(zt1, zt1_hat)**2)
        L_trans = torch.mean(torch.sum((zt1 - zt1_hat) ** 2, -1))

        return L_qm, L_trans

    def actor_loss(self, st, goal_t, st1, at):

        goal = torch.cat([goal_t, st[:, goal_t.shape[1]:]], -1)
        z = self.f(st)
        z_g = self.f(goal)
        a = torch.tanh(self.pi(z.detach(), goal_t.detach()))
        neg_Q = self.get_distance(self.T(z.detach(), a), z_g.detach())
        tot_loss = torch.mean(neg_Q) + 0*torch.mean((a - at)**2)

        # log_prob = self.pi.get_log_prob(st, at, goal_t)
        # Vt = self.get_value(st, goal_t)
        # Vt1 = self.get_value(st1, goal_t)
        # Adv = (Vt1 - Vt).detach()
        # Adv_skew = torch.exp(Adv * self.R_gamma)-1
        # tot_loss = - torch.mean(Adv_skew * log_prob)

        return tot_loss

    def update(self, batch=None, epoch=None, train_modules=None):

        st, gt, at, rt, _, st1 = batch

        st = st if st.is_cuda else st.to(self.device)
        gt = gt if gt.is_cuda else gt.to(self.device)
        at = at if at.is_cuda else at.to(self.device)
        st1 = st1 if st1.is_cuda else st1.to(self.device)

        L_qm, L_trans = self.critic_loss(st, gt, at, rt, st1)

        critic_loss = L_qm + L_trans

        self.opt_critic.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        self.opt_critic.step()

        L_pi = self.actor_loss(st, gt, st1, at)

        self.opt_actor.zero_grad()
        L_pi.backward()
        # torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        self.opt_actor.step()

        logs_losses = {'L_pos': L_qm.detach().cpu().item() if L_qm is not None else 0,
                       'L_neg': 0,
                       'L_trans': L_trans.detach().cpu().item() if L_trans is not None else 0,
                       'L_pi': L_pi.detach().cpu().item() if L_pi is not None else 0,
                       'L_DQN': 0}

        return logs_losses

