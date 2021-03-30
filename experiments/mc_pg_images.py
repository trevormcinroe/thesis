"""
static_static_gray_overcam -- 249

"""
import time
import argparse
import torch
from torch import optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import pickle
import numpy as np
from collections import namedtuple
from environments.kuka import KukaEnv
from models import PolicyCNN, PolicyFC
from utils import make_pca_plot

parser = argparse.ArgumentParser(description='Training simulation for various deep RL environments.')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--seed', type=int, default=19, help='random seed')
parser.add_argument('--repeat', type=int, default=25, help='number of times to repeat a given action')
parser.add_argument('--max-ep-len', type=int, default=1000, help='maximum number of steps per episode')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how often to print progress')
parser.add_argument('--episodes', type=int, default=10000, help='number of training episodes to run')
parser.add_argument('--images', action='store_true', default=False, help='Image-based states')
parser.add_argument('--name', help='experiment name')
args = parser.parse_args()

env = KukaEnv(
    renders=args.render,
    is_discrete=True,
    max_steps=args.max_ep_len,
    action_repeat=args.repeat,
    images=args.images,
    static_all=True,
    static_obj_rnd_pos=False,
    rnd_obj_rnd_pos=False,
    full_color=True
)

env.seed(args.seed)
torch.manual_seed(args.seed)

saved_action = namedtuple('saved_action', ['log_prob', 'value'])

if args.images:
    model = PolicyCNN()
    model.load_state_dict(torch.load('./models/10_fewshot_0.pth'))
    # Trying out freezing some weights
    # model.c1.weight.requires_grad = False
    # model.c1.bias.requires_grad = False
    # model.c2.weight.requires_grad = False
    # model.c2.bias.requires_grad = False
    # model.c3.weight.requires_grad = False
    # model.c3.bias.requires_grad = False
    # model.d1.weight.requires_grad = False
    # model.d1.bias.requires_grad = False
    # print(model.state_dict())

else:
    model = PolicyFC()

optimizer = optim.Adam(model.parameters())
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    """

    Args:
        state:

    Returns:

    """

    if env.images:
        state = transforms.ToTensor()(state).unsqueeze(0).float()
        probs, state_value = model(state)
        probs = probs[0]
        state_value = state_value[0]
    else:
        state = torch.from_numpy(state).float()

        probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(saved_action(m.log_prob(action), state_value))

    # the action to take
    return action.item()


def finish_episode():
    """
    Training code. Calcultes actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():

    start = time.time()

    running_reward = 10

    completed = []

    # run inifinitely many episodes
    # for i_episode in count(1):
    for i in range(args.episodes):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        inner_picked_up = []
        # for each episode, only run some steps so that we don't
        # infinite loop while learning
        for t in range(1, args.max_ep_len):
            action = select_action(state)
            state, reward, picked_up, done, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            inner_picked_up.append(picked_up)

            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # Logging the picked up
        if np.sum(inner_picked_up) > 0:
            completed.append(1)
        else:
            completed.append(0)

        # log results
        if i % args.log_interval == 0:
            print(f'Episode {i}, Pct: {np.mean(completed[-100:])}, Hours time {(time.time() - start) / 3600}')
            # print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tPct: {:.4f}\'.format(
            #       i, ep_reward, running_reward, np.mean(completed[-100:])))

        if i % 2500 == 0:
            title = f'{i}: PCA'
            save_name = f'./results/{i}_{args.name}.png'
            make_pca_plot(model, './full_states_0.data', './rewards.data', title, save_name)
        #     filename_net = 'results/exp_model_' + str(args.name) + f'_{i}.pth'
        #     torch.save(model.state_dict(), filename_net)

        # check if we have "solved" the problem
        # if running_reward > args.aver: #env.spec.reward_threshold:
    print("Run has completed!")
    checkpoint = {
        'episode': i,
        'state_dict': model.state_dict(),
        'ep_reward': ep_reward,
        'running_reward': running_reward
    }
    # filename_net = 'model_aver_' + str(args.aver) + '.pth'
    filename_net = 'results/model_' + str(args.name) + '.pth'
    torch.save(model.state_dict(), filename_net)

    # Plotting
    with open(f'./results/results_{args.name}.data', 'wb') as file:
        pickle.dump(completed, file)

    smoothed = []
    b_idx = 0
    e_idx = 100
    while e_idx < len(completed):
        smoothed.append(np.mean(completed[b_idx:e_idx]))
        b_idx += 1
        e_idx += 1
    fig = plt.figure(dpi=400)
    plt.plot(smoothed)
    plt.title('Success Rate in Kuka Environment (Pick Up)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(0., 1.)
    plt.savefig(f'./results/results_{args.name}.png')

    print(f'Experiment {args.name} completed. \n Total time (hrs): {(time.time() - start) / 3600}')


if __name__ == '__main__':
    main()
