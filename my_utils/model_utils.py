import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
import random
from tqdm import tqdm


class Memory(object):

    def __init__(self, size):
        self.size = size
        self.memory = []
        self.position = 0

    def push(self, st, obst, gt, a, a_idx, r, terminal, st1):
        if len(self.memory) < self.size:
            self.memory.append(None)

        element = {'st': st, 'obst': obst, 'goal_t': gt, 'a': a, 'a_idx': a_idx, 'r': r, 'terminal': terminal, 'st1': st1}

        self.memory[int(self.position)] = element
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def fill_up_actions(all_actions, current_actions, max_depth, depth, i):
    if depth == max_depth:
        all_actions[i] = current_actions
        return all_actions, i+1

    for value in [-1., 0., 1.]:
        current_actions[depth] = value
        all_actions, i = fill_up_actions(all_actions, current_actions, max_depth, depth+1, i)

    return all_actions, i


def fill_up_transitions(env, all_transitions, current_state, current_depth):
    if current_depth == env.dim:
        if env.check_if_obst(current_state):
            return all_transitions
        for a_idx, a in enumerate(env.all_actions):
            neigh_obst = env.get_surrounding(current_state)
            step = {'s': current_state.copy(), 'obst': neigh_obst, 'g': env.goal, 'a': a, 'a_idx': a_idx}
            next_obs, reward, term, _, _ = env.step(a, state=current_state.copy())
            step['r'] = reward
            step['term'] = (not term) * 1.
            step['s1'] = next_obs['observation'].copy()
            all_transitions.append(step)
        return all_transitions

    if current_depth == 0:
        for x in tqdm(np.arange(0, 1, env.step_size)):
            current_state[current_depth] = x
            all_transitions = fill_up_transitions(env, all_transitions, current_state, current_depth+1)
    else:
        for x in np.arange(0, 1, env.step_size):
            current_state[current_depth] = x
            all_transitions = fill_up_transitions(env, all_transitions, current_state, current_depth+1)

    return all_transitions


class Residual_Block(nn.Module):

    def __init__(self, in_dim):
        super().__init__()

        hidden = 64
        self.f = nn.Sequential(nn.Linear(in_dim, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, in_dim))

    def forward(self, x):
        h = self.f(x)
        return h + x

class MLP(nn.Module):

    def __init__(self, input_dim, cond_dim, output_dim, n_layers, residual=False):
        super().__init__()

        hidden = 64

        net_input = input_dim
        if cond_dim > 0:
            self.g_proj = nn.Sequential(nn.Linear(cond_dim, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, hidden)) # input_dim
            net_input += hidden#input_dim

        self.f = nn.ModuleList()
        self.f.append(nn.Linear(net_input, hidden))
        self.f.append(nn.ReLU())
        if residual:
            for _ in range(n_layers):
                self.f.append(Residual_Block(hidden))
        else:
            for _ in range(n_layers):
                self.f.append(nn.Linear(hidden, hidden))
                self.f.append(nn.ReLU())
        self.f.append(nn.Linear(hidden, output_dim))

    def forward(self, x, cond=None):

        if cond is not None:
            c = self.g_proj(cond)
            h = torch.cat([x, c], -1)
        else:
            h = x

        for layer in self.f:
            h = layer(h)
        return h

class CNN(nn.Module):

    def __init__(self, channels, cond_dim, output_dim):
        super().__init__()

        hidden = 64
        flat_dims = hidden * 8 * 8 #6 * 6
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),

        )

        self.head = nn.Sequential(
            nn.Linear(flat_dims+cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x, c=None):
        h = self.body(x)
        h = h if c is None else torch.cat([h, c], -1)
        return self.head(h)


class Policy(nn.Module):

    def __init__(self, input_dim, cond_dim, output_dim, policy_def):
        super().__init__()

        n_layers = 2 #3
        hidden_units = 32 #256 #128

        self.type = policy_def['type']
        self.var = policy_def['var']
        self.max_action = policy_def['max_action']
        self.bias_action = policy_def['bias_action']

        net_input = input_dim
        # if cond_dim > 0:
        #     self.g_proj = nn.Sequential(nn.Linear(cond_dim, hidden_units),
        #                                 nn.ReLU(),
        #                                 nn.Linear(hidden_units, hidden_units),
        #                                 nn.ReLU(),
        #                                 nn.Linear(hidden_units, input_dim))
        #     net_input += input_dim
        self.g_proj = None
        net_input += cond_dim

        self.f = nn.ModuleList()
        self.f.append(nn.Linear(net_input, hidden_units))
        self.f.append(nn.ReLU())
        for _ in range(n_layers):
            self.f.append(nn.Linear(hidden_units, hidden_units))
            self.f.append(nn.ReLU())
        self.f.append(nn.Linear(hidden_units, output_dim))

    def get_mean(self, x, c=None):

        if c is not None:
            if self.g_proj is not None:
                c = self.g_proj(c)
            h = torch.cat([x, c], -1)
        else:
            h = x

        for layer in self.f:
            h = layer(h)

        mu = torch.softmax(h, -1) if self.type == 'discrete' else torch.tanh(h) * self.max_action + self.bias_action

        return mu

    def get_log_prob(self, x, a, c=None):

        mu = self.get_mean(x, c)
        dist = Categorical(probs=mu) if self.type == 'discrete' else Normal(loc=mu, scale=self.var)
        return dist.log_prob(a[:,0]) if self.type == 'discrete' else dist.log_prob(a).sum(-1), dist.entropy() # N.log_prob(a[:,0]), N.entropy()

    def sample_action(self, x, c=None):
        mu = self.get_mean(x, c)
        dist = Categorical(probs=mu) if self.type == 'discrete' else Normal(loc=mu, scale=self.var)
        return dist.sample().float()


class Policy_discrete(nn.Module):

    def __init__(self, input_dim, cond_dim, output_dim, network_def):
        super().__init__()

        n_layers = network_def['n_layers']
        hidden_units = network_def['hidden_units']
        activation = nn.ReLU() if network_def['activation'] == 0 else nn.Tanh()
        projection = network_def['projection'] == 1

        net_input = input_dim
        if projection == 1:
            self.g_proj = nn.Sequential(nn.Linear(cond_dim, hidden_units),
                                        activation,
                                        nn.Linear(hidden_units, hidden_units),
                                        activation,
                                        nn.Linear(hidden_units, input_dim))
            net_input += input_dim
        else:
            self.g_proj = None
            net_input += cond_dim

        self.f = nn.ModuleList()
        self.f.append(nn.Linear(net_input, hidden_units))
        self.f.append(activation)
        for _ in range(n_layers):
            self.f.append(nn.Linear(hidden_units, hidden_units))
            self.f.append(activation)
        self.f.append(nn.Linear(hidden_units, output_dim))

    def get_mean(self, x, c=None):

        if c is not None:
            if self.g_proj is not None:
                c = self.g_proj(c)
            h = torch.cat([x, c], -1)
        else:
            h = x

        for layer in self.f:
            h = layer(h)

        #mu = torch.tanh(h) * self.max_action + self.bias_action
        mu = torch.softmax(h, -1)

        return mu

    def get_log_prob(self, x, a, c=None):

        mu = self.get_mean(x, c)
        N = Categorical(probs=mu) #logits
        return N.log_prob(a[:,0]), N.entropy()

    def sample_action(self, x, c=None):
        mu = self.get_mean(x, c)
        N = Categorical(probs=mu)
        return N.sample()


class ActorCritic(nn.Module):

    def __init__(self, input_dim, a_dim, a_max, init_layers, learn_var):
        super(ActorCritic, self).__init__()

        self.learn_var = learn_var
        self.a_dim = a_dim
        self.a_max = a_max
        hidden_fc = 64

        self.actor_head = nn.Sequential(nn.Linear(input_dim, hidden_fc),
                                        nn.ReLU(),
                                        nn.Linear(hidden_fc, hidden_fc),
                                        nn.ReLU(),
                                        nn.Linear(hidden_fc, hidden_fc),
                                        nn.ReLU())
        self.actor_mu = nn.Sequential(nn.Linear(hidden_fc, hidden_fc),
                                      nn.ReLU(),
                                      nn.Linear(hidden_fc, a_dim))
        if learn_var:
            self.actor_sigma = nn.Sequential(nn.Linear(hidden_fc, hidden_fc),
                                             nn.ReLU(),
                                             nn.Linear(hidden_fc, a_dim))
        else:
            self.var = nn.parameter.Parameter(torch.ones(1, a_dim) * 0.3, requires_grad=False).float()

        self.value = nn.Sequential(nn.Linear(input_dim, hidden_fc),
                                   nn.ReLU(),
                                   nn.Linear(hidden_fc, hidden_fc),
                                   nn.ReLU(),
                                   nn.Linear(hidden_fc, hidden_fc),
                                   nn.ReLU(),
                                   nn.Linear(hidden_fc, 1))

        #torch.nn.init.xavier_uniform_(self.actor_head.weight, gain=nn.init.calculate_gain('linear', self.film3.weight))
        #nn.init.orthogonal_(w)

        if init_layers:
            torch.nn.init.orthogonal_(self.actor_head[0].weight)
            torch.nn.init.orthogonal_(self.actor_head[2].weight)
            torch.nn.init.orthogonal_(self.actor_head[4].weight)
            torch.nn.init.orthogonal_(self.actor_mu[0].weight)
            torch.nn.init.orthogonal_(self.actor_mu[2].weight)
            # torch.nn.init.orthogonal_(self.actor_sigma[0].weight)
            # torch.nn.init.orthogonal_(self.actor_sigma[2].weight)

    def forward(self, st):

        policy_h = self.actor_head(st)
        v = self.value(st)
        mu = torch.tanh(self.actor_mu(policy_h)) * self.a_max
        if self.learn_var:
            sigma = torch.sigmoid(self.actor_sigma(policy_h)) * 10.0 + 0.0001 # TODO: change me
        else:
            sigma = self.var #torch.sigmoid(self.actor_sigma(policy_h)) * 1.0 + 0.0001
        # mu = torch.tanh(policy[:, 0:self.a_dim]) * self.a_max
        # sigma = torch.sigmoid(policy[:, self.a_dim:2 * self.a_dim]) * 1 + 0.0001
        v = v[:, -1]

        return mu, sigma, v

    def get_action(self, st, test=False):

        mu, sigma, v = self.forward(st)

        if test:
            return mu, None, None

        m = MultivariateNormal(mu, torch.diag_embed(sigma))
        a = m.sample()
        logprob = m.log_prob(a)
        H = m.entropy()

        return a, logprob, sigma.detach().cpu().numpy()

    def evaluate(self, st, at):

        mu, sigma, v = self.forward(st)

        m = MultivariateNormal(mu, torch.diag_embed(sigma))
        logprob = m.log_prob(at)
        H = m.entropy()

        return logprob, v, H

class Network_policy(nn.Module):

    def __init__(self, state_dim, action_dim, device):
        super(Network_policy, self).__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
        )

        self.policy_mean = nn.Sequential(
            nn.Linear(256, action_dim, bias=True)
        )

        self.policy_var = nn.Sequential(
            nn.Linear(256, action_dim, bias=True),
            nn.Sigmoid(),
        )

        self.N = torch.distributions.normal.Normal(torch.zeros(action_dim).to(device), torch.ones(action_dim).to(device))

    def get_action(self, s, greedy=False):
        if greedy:
            return torch.tanh(self.policy_mean(self.body(s))), None
        h = self.body(s)
        mean = self.policy_mean(h)
        var = torch.clip(self.policy_var(h), 1e-10, 2.)
        xi = self.N.sample()
        u = mean + var * xi.detach()
        a = torch.tanh(u)
        # neg_log_pi = torch.sum(F.gaussian_nll_loss(mean, a, var, reduction='none'), -1) + torch.sum(torch.log(1 - torch.tanh(a)**2), -1)
        log_pi = torch.sum(-torch.log(torch.clamp(var, 1e-6)), -1) - 0.5 * torch.sum(((u - mean) / var)**2, -1) - torch.sum(torch.log(1 - torch.tanh(u)**2), -1)
        return a, log_pi

#
#
# def test_render(model, device, max_T=300, sim_episodes=10):
#     sim = gym.make('PointMaze_UMaze-v3', max_episode_steps=max_T, continuing_task=True, render_mode='human')  # human
#     model.eval()
#     for i in range(sim_episodes):
#         obs, _ = sim.reset()
#         current_goal = to_tensor(obs['desired_goal'], device=device)
#         current_state = to_tensor(obs['observation'], device=device)
#         for t in range(max_T):
#             sim.render()
#             current_action = model.get_action(current_state, current_goal)
#             obs, reward, terminated, truncated, info = sim.step(current_action)
#             current_goal = to_tensor(obs['desired_goal'], device=device)
#             current_state = to_tensor(obs['observation'], device=device)  # TODO: if I reach the goal does the episode end or continues with a different goal ???
#             if terminated or truncated:
#                 break
#
#
# def test_pos_val(model, dataset, device):
#     all_episodes = dataset.sample_episodes(1000)
#     all_states = to_tensor(np.concatenate([np.concatenate((episode.observations['observation'][:-1, :2],
#                                                            np.zeros((episode.observations['observation'].shape[0]-1, 2))), -1) for episode in all_episodes], 0), device=device)
#     V = model.get_value(all_states, torch.tensor([[-1., -1.]]).to(device))
#     return all_states.detach().cpu().numpy(), V.detach().cpu().numpy()
#     # ax = plt.axes(projection='3d')
#     # ax.plot_trisurf(s[:,0], s[:,1], Vn, cmap='viridis', edgecolor='none')
#     # ax.view_init(elev=10, azim=angle)
#     # plt.show()
#
# # grid = np.zeros((100, 100))
# # for i, x in enumerate(np.linspace(-1.45, 1.45, 100)):
# #     for j, y in enumerate(np.linspace(-1.45, 1.45, 100)):
# #         neibors = (np.abs(s[:,0]-x) < (1.45*2/100))*(np.abs(s[:,1]-y) < (1.45*2/100))*1.
# #         if np.sum(neibors*1) > 0:
# #             all_val_idx = np.nonzero(neibors)
# #             grid[i, j] = np.max(Vn[all_val_idx])
#
# def test_vel_val(model, dataset, device):
#     all_episodes = dataset.sample_episodes(1000)
#
#     all_states = to_tensor(np.concatenate([episode.observations['observation'][:-1] for episode in all_episodes], 0), device=device)
#     min_vel, max_vel = torch.min(all_states[:, 2:]).detach().cpu().item(), torch.max(all_states[:, 2:]).detach().cpu().item()
#     start_state = torch.tensor([[-1.0, 1.0, 1.0, 0.0]]).float().to(device)
#     end_state = torch.tensor([[1.0, 1.0]]).float().to(device)
#     all_vel_vals = torch.arange(min_vel, max_vel, 0.01)
#     all_vel = all_vel_vals.view(-1, 1).repeat(1, 4).to(device)
#     all_vel[:, :2] = all_vel[:, :2] * 0 + 1
#     all_vel[:, 3] = all_vel[:, 3] * 0
#     all_vel_states = all_vel * start_state
#
#     V = model.get_value(all_vel_states, end_state)
#
#     return all_vel_vals.detach().cpu().numpy(), V.detach().cpu().numpy()
#
#
# def test_isomap(model, dataset, device):
#     all_episodes = dataset.sample_episodes(100)
#     all_states = to_tensor(np.concatenate([episode.observations['observation'][:-1] for episode in all_episodes], 0), device=device)
#     z_space = model.get_state_repr(all_states)
#     embedding = Isomap(n_neighbors=15, n_components=2)
#     X_transformed = embedding.fit_transform(z_space.detach().cpu().numpy())
#
#     s_space = all_states.detach().cpu().numpy()
#     min_x, max_x, min_y, max_y = np.min(s_space[:, 0]), np.max(s_space[:, 0]), np.min(s_space[:, 1]), np.max(s_space[:, 1])
#     x_norm = (s_space[:, 0] - min_x) / (max_x - min_x)
#     y_norm = (s_space[:, 1] - min_y) / (max_y - min_y)
#     color = np.zeros((x_norm.shape[0], 3))
#     color[:, 1] = x_norm
#     color[:, 2] = y_norm
#
#     return X_transformed, color