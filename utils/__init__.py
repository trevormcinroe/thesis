import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
from torchvision.transforms import ToTensor
from torch.distributions import Categorical
import gym
from skimage.util.shape import view_as_windows


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def inspect_episode(agent, env, device, experiment_name, step):

    info = {
        's': [],
        'a': [],
        'r': [],
        's_': [],
        't': [],
        'aprobs': []
    }
    s = env.reset()

    done = False

    while not done:
        a, aprobs = agent.sample_action(s.unsqueeze(0).float().to(device), inspect=True)

        s_, r, picked_up, t, _ = env.step(a)

        if t:
            done = True

        info['s'].append(s)
        info['a'].append(a)
        info['r'].append(r)
        info['s_'].append(s_)
        info['t'].append(t)
        info['aprobs'].append(aprobs)

        s = s_

    with open(f'./{experiment_name}-{step}.data', 'wb') as f:
        pickle.dump(info, f)




def make_pca_plot(model, states, scaled_rewards, title, save_name):
    """ Helper function to plot the internal representation of a model
    using PCA.

    Args:
        model:
        states:
        scaled_rewards:
        title:
        save_name:

    Returns:
        None
    """
    outs = []

    with open(states, 'rb') as f:
        fulls = pickle.load(f)

    with open(scaled_rewards, 'rb') as f:
        scaled_rewards = pickle.load(f)

    scaled_rewards = [x if x < 10 else 1 for x in scaled_rewards[0]]

    for i in range(len(fulls)):
        img = np.moveaxis(fulls[i], -1, 0)
        img = torch.tensor(img).unsqueeze(0).float()
        out = model.base(img)
        outs.append(out.detach().numpy()[0])

    pca = PCA(3)
    x = pca.fit_transform(np.array(outs))


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title(title)
    ax.scatter(xs=x[:, 0],
               ys=x[:, 1],
               zs=x[:, 2],
               c=scaled_rewards, cmap='winter')
    plt.savefig(save_name)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    pass
        # if isinstance(m, nn.Linear):
        #     nn.init.orthogonal_(m.weight.data)
        #     m.bias.data.fill_(0.0)
        # elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #     # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        #     assert m.weight.size(2) == m.weight.size(3)
        #     m.weight.data.fill_(0.0)
        #     m.bias.data.fill_(0.0)
        #     mid = m.weight.size(2) // 2
        #     gain = nn.init.calculate_gain('relu')
        #     nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

# def weight_init(m):
#     pass
#     # if isinstance(m, nn.Linear):
#     #     torch.nn.init.xavier_uniform_(m.weight, gain=1)
#     #     torch.nn.init.constant_(m.bias, 0)


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


class FrameStackEnv:
    def __init__(self, env, k, data_type='np', state_type='images'):
        self.frames = None
        self.env = env
        self.k = k
        self.data_type = data_type
        self.state_type = state_type

        self.reset()

    @property
    def images(self):
        return self.env.images

    def reset(self):
        if self.state_type == 'images':
            if self.data_type == 'np':
                self.frames = deque([self.env.reset() for _ in range(self.k)], maxlen=self.k)
            else:
                self.frames = deque([ToTensor()(self.env.reset()) for _ in range(self.k)], maxlen=self.k)

        else:
            if self.data_type == 'np':
                self.frames = deque([self.env.reset() for _ in range(self.k)], maxlen=self.k)
            else:
                self.frames = deque([torch.tensor(self.env.reset()) for _ in range(self.k)], maxlen=self.k)

        return self._get_obs()

    def step(self, action):
        state, reward, picked_up, done, _ = self.env.step(action)

        if self.state_type == 'images':
            self.frames.append(ToTensor()(state))
        else:
            self.frames.append(torch.tensor(state))

        return self._get_obs(), reward, picked_up, done, _

    def _get_obs(self):
        if self.data_type == 'np':
            return np.concatenate(list(self.frames), axis=2)
        else:
            return torch.cat(list(self.frames), axis=0)


class ExperienceGenerator:
    def __init__(self, env):
        self.env = env

    def play_n_episodes(self, episodes_per_learning_round, actor):
        """

        Args:
            episodes_per_learning_round:
            actor:

        Returns:
            c_states, c_actions, c_rewards List[Lists] -- c_completed List[ints] -- global_steps int
        """
        collected_states = []
        collected_actions = []
        collected_rewards = []
        collected_completed = []
        global_steps = 0

        for _ in range(episodes_per_learning_round):
            states, actions, rewards, completed, steps = self.play_1_episode(actor)
            collected_states.append(states)
            collected_actions.append(actions)
            collected_rewards.append(rewards)
            if np.sum(completed) > 0:
                collected_completed.append(1)
            else:
                collected_completed.append(0)
            global_steps += steps

        return collected_states, collected_actions, collected_rewards, collected_completed, global_steps

    def play_1_episode(self, actor):
        s = self.env.reset()

        states = []
        actions = []
        rewards = []
        completed = []

        steps = 0

        done = False

        while not done:
            actor_output = actor(s.float())
            action_dist = Categorical(actor_output)
            a = action_dist.sample()
            s_, r, picked_up, t, _ = self.env.step(a.item())

            states.append(s)
            actions.append(a.item())
            rewards.append(r)
            completed.append(picked_up)

            s = s_

            steps += 1

            if t:
                done = True
            if picked_up:
                done = True

        return states, actions, rewards, completed, steps


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def sanity_check(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.all(p1.eq(p2)):
            return False
    return True


def eval_agent(env, agent, num_episodes):
    collected_rs = []

    for _ in range(num_episodes):
        s = env.reset()
        s = torch.tensor(s / 255.).float()
        episode_r = 0
        done = False

        while not done:
            with eval_mode(agent):
                a = agent.select_action(s.unsqueeze(0).to('cuda:0'))
            s, r, done, _ = env.step(a)
            s = torch.tensor(s / 255.).float()
            episode_r += r

        collected_rs.append(episode_r)

    return np.mean(collected_rs)


def eval_representation(model, replay_memory):
    err_hist = []

    W = torch.nn.parameter.Parameter(torch.rand(1, 50)).to('cuda:0')
    torch.nn.init.normal_(W)
    W_optim = torch.optim.Adam([W.data], lr=1e-4)

    for _ in range(50):
        s, _, r, _, _ = replay_memory.sample(256)
        out = torch.matmul(model(s.to('cuda')).detach(), W.T)
        loss = F.mse_loss(out, r.unsqueeze(-1).to('cuda:0'))
        W_optim.zero_grad()
        loss.backward()
        W_optim.step()

    for _ in range(100):
        s, _, r, _, _ = replay_memory.sample(256)
        out = torch.matmul(model(s.to('cuda')).detach(), W.T)
        loss = F.mse_loss(out, r.unsqueeze(-1).to('cuda:0'))
        err_hist.append(loss.item())

    return np.mean(err_hist)


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones
    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

def center_crop_image(image, output_size):
    """images are coming in [b, c, h, w]"""
    h, w = image.shape[2:]

    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    # we want to keep b anc c...
    image = image[:, :, top:top + new_h, left:left + new_w]
    return image


def byol_loss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def r_proj_loss(z_online, z_target, r_online, r_target):
    z_online = F.normalize(z_online, dim=-1, p=2)
    z_target = F.normalize(z_target, dim=-1, p=2)

    z_dist = 2 - 2 * (z_online * z_target).sum(dim=-1)

    r_dist = torch.abs_(r_online - r_target).squeeze()

    return (z_dist - r_dist).mean()
