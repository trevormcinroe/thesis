import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyFC(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, s_dim):
        super(PolicyFC, self).__init__()
        self.affine1 = nn.Linear(s_dim, 32)
        self.affine2 = nn.Linear(32, 64)

        # actor's layer
        self.action_head = nn.Linear(64, 7) # reduced action Point EC

        # critic's layer
        self.value_head = nn.Linear(64, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


# class PolicyCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c1 = nn.Conv2d(3, 8, (3, 3), stride=2)
#         self.c2 = nn.Conv2d(8, 4, (3, 3), stride=2)
#         self.c3 = nn.Conv2d(4, 1, (3, 3), stride=2)
#         self.d1 = nn.Linear(729, 64)
#
#         self.action_head = nn.Linear(64, 7)
#         self.value_head = nn.Linear(64, 1)
#
#         # action & reward buffer
#         self.saved_actions = []
#         self.rewards = []
#
#     def forward(self, x):
#         x = F.relu(self.c1(x))
#         x = F.relu(self.c2(x))
#         x = F.relu(self.c3(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.d1(x))
#
#
#         # actor: choses action to take from state s_t
#         # by returning probability of each action
#         action_prob = F.softmax(self.action_head(x), dim=-1)
#
#         # critic: evaluates being in the state s_t
#         state_values = self.value_head(x)
#
#         # return values for both actor and critic as a tupel of 2 values:
#         # 1. a list with the probability of each action over the action space
#         # 2. the value from state s_t
#         return action_prob, state_values


class DQNCnn(nn.Module):
    """Meant to function with frame-stacking!"""
    def __init__(self, n_actions):
        super().__init__()
        self.c1 = nn.Conv2d(4, 8, (5, 5), stride=2)
        self.c2 = nn.Conv2d(8, 4, (3, 3), stride=2)
        self.c3 = nn.Conv2d(4, 1, (3, 3), stride=2)
        self.d1 = nn.Linear(676, n_actions)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = x.view(x.size(0), -1)
        return self.d1(x)


class PolicyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, (3, 3), stride=2)
        self.c2 = nn.Conv2d(8, 4, (3, 3), stride=2)
        self.c3 = nn.Conv2d(4, 1, (3, 3), stride=2)
        self.d1 = nn.Linear(729, 64)

        self.action_head = nn.Linear(64, 7)
        self.value_head = nn.Linear(64, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def base(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = x.view(x.size(0), -1)
        return self.d1(x)

    def siamese(self, xs):
        out1 = self.base(xs[0])
        out2 = self.base(xs[1])
        return out1, out2

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.d1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

