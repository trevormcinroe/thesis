import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch


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
