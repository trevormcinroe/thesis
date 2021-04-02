import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf. (SAC PAPER)
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