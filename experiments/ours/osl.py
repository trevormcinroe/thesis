"""
We normalize activations to lie in [0,1] at the output of the convolutional encoder and transition model,
as in Schrittwieser et al. (2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.nn.utils import clip_grad_norm_
import utils
import hydra

DROPOUT = 0.0
DROPOUT_FC = 0.0

class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = True
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.Dropout(DROPOUT),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Dropout(DROPOUT),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Dropout(DROPOUT),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.Dropout(DROPOUT)
        ])

        self.head = nn.Sequential(
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.outputs = dict()

    def forward_conv(self, obs, detach=False):
        conv = obs / 255.
        # self.outputs['obs'] = obs

        # conv = torch.relu(self.convs[0](obs))
        # self.outputs['conv1'] = conv

        for layer in self.convs:
            if 'stride' not in layer.__constants__:
                conv = layer(conv)
            else:
                conv = torch.relu(layer(conv))

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()
        out = self.head(h)
        if not self.output_logits:
            # out = torch.sigmoid(out)
            out = torch.tanh(out)

        self.outputs['out'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        for i in range(len(self.convs)):
            if 'stride' not in self.convs[i].__constants__:
                pass
            else:
                utils.tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, logger, step):
        pass
        # for k, v in self.outputs.items():
        #     logger.log_histogram(f'train_encoder/{k}_hist', v, step)
        #     if len(v.shape) > 2:
        #         logger.log_image(f'train_encoder/{k}_img', v[0], step)
        #
        # for i in range(self.num_layers):
        #     logger.log_param(f'train_encoder/conv{i + 1}', self.convs[i], step)


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim,
                               2 * action_shape[0], hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def noise(self, obs, n, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        obs[0][np.random.choice(range(len(obs[0])), n, replace=False)] = 0

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, encoder_cfg, action_shape, hidden_dim, hidden_depth):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder_cfg)

        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0],
                            hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)


        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        self.encoder.log(logger, step)

        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class DenseTrans(nn.Module):
    def __init__(self, critic, large_overlap=False):
        super().__init__()

        if large_overlap:
            self.q1 = critic.Q1[:4]
            self.q2 = critic.Q2[:4]


            self.dense_head = nn.ModuleList([
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(DROPOUT_FC),
                nn.Linear(512, 50)
            ])

        else:

            # For knowledge-sharing ablation
            self.q1 = critic.Q1[0:1]
            self.q2 = critic.Q2[0:1]
            #
            # self.q1 = nn.Linear(critic.Q1[0].in_features, critic.Q1[0].out_features)
            # self.q2 = nn.Linear(critic.Q2[0].in_features, critic.Q2[0].out_features)


            # For huge overlap ablation

            self.dense_head = nn.ModuleList([
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(DROPOUT_FC),
                nn.Linear(512, 50)
            ])

        self.apply(utils.weight_init)

    def forward(self, h, a):
        """Takes the hidden feature maps from an encoder and outputs <B x 32 x 8 x 8> or <B x 2048>
        """
        h = torch.cat([h, a], dim=-1)

        h1 = self.q1(h)
        h2 = self.q2(h)

        h = (h1 + h2) / 2

        for layer in self.dense_head:
            h = layer(h)

        return h


class ProjectionHead(nn.Module):
    """Uses the first two layers of the critics before data enters here..."""
    def __init__(self, ):
        super().__init__()

        self.proj_head = nn.ModuleList([
            nn.Linear(50, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_FC),
            nn.Linear(512, 50),
        ])

        self.apply(utils.weight_init)

    def forward(self, x):

        for layer in self.proj_head:
            x = layer(x)
        return x

class RewardPredictor(nn.Module):
    """Rewards are not [0,1] due to frame-skipping..."""
    def __init__(self, z_dim, action_shape):
        super().__init__()
        print(action_shape)
        self.net = nn.Sequential(
            nn.Linear(z_dim + action_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z, a):
        z = torch.cat([z, a], dim=1)
        r_hat = self.net(z)
        return r_hat


class OSL(nn.Module):
    def __init__(self, critic_online, critic_momentum, action_shape):
        """

        Encoders: <B x 50>
        Trans: <B x 50>

        """
        super().__init__()
        self.encoder_online = critic_online.encoder
        self.encoder_momentum = critic_momentum.encoder

        self.transition_model = DenseTrans(critic_online, False)

        self.proj_online = ProjectionHead()
        self.proj_momentum = ProjectionHead()

        self.proj_momentum.load_state_dict(self.proj_online.state_dict())

        self.Wz = nn.Linear(50, 50, bias=False)

        self.Wsingle = nn.Linear(50 + action_shape, 50)

        self.Wr = RewardPredictor(50, action_shape)
        # self.Wr = nn.Linear(50, 1, bias=False)

    def encode(self, s, s_):
        h = self.encoder_online(s)
        h_ = self.encoder_momentum(s_).detach()

        return h, h_

    def transition(self, h, a):
        h = self.transition_model(h, a)
        return h

    def projection(self, h, h_):
        projection = self.proj_online(h)
        projection_ = self.proj_momentum(h_).detach()

        return projection, projection_

    def predict(self, projection):
        z_hat = self.Wz(projection)

        return z_hat


class DRQAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, obs_shape, action_shape, action_range, device,
                 encoder_cfg, critic_cfg, actor_cfg, discount,
                 init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size, osl_update_frequency, k):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.osl_update_frequency = osl_update_frequency
        self.k = k

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Orig places ALL tensors into optimizer (including encoder)
        # let's remove this. I think the RL objective is obfuscating with repr learning
        # THIS TEST DID NOT WORK
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        # self.critic_optimizer = torch.optim.Adam(
        #     list(self.critic.Q1.parameters()) + list(self.critic.Q2.parameters()),
        #     lr=lr
        # )

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train()
        self.critic_target.train()

        self.osl = OSL(self.critic, self.critic_target, action_shape[0]).to(self.device)
        self.osl_optimizer = torch.optim.Adam(self.osl.parameters(), lr=1e-4)
        self.byol_loss = byol_loss
        self.spr_loss = spr_loss

        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=lr
        )

        # print(self.osl)

        self.osl_loss_hist = []
        self.r_loss_hist = []

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train(training)
        # self.osl.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def act_noise(self, obs, n, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor.noise(obs, n=n)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])


    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, logger, step):
        """Target Q seems to be more important"""
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                  keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                      next_action_aug)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        logger.log('train_alpha/loss', alpha_loss, step)
        logger.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_osl(self, obs, a, next_obs):
        self.osl.train(True)

        # h, h_ = self.osl.encode(obs, next_obs)
        #
        # h = self.osl.transition(h, a)
        #
        # projection, projection_ = self.osl.projection(h, h_)
        #
        # projection_hat = self.osl.predict(projection)
        #
        # loss = self.byol_loss(projection_hat, projection_).mean() * 2
        #
        # # loss = self.spr_loss(projection_hat, projection_) * 2
        # self.osl_loss_hist.append(loss.item())
        # print(loss.item())

        z_ = self.osl.encoder_momentum(next_obs).detach()

        z = self.osl.encoder_online(obs)
        z_hat = self.osl.Wsingle(torch.cat([z, a]))

        loss = self.byol_loss(z_hat, z_).mean()

        self.osl_loss_hist.append(loss.item())

        self.osl_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        loss.backward()

        # clip_grad_norm_(self.osl.parameters(), 10)
        # clip_grad_norm_(self.critic.encoder.parameters(), 10)

        self.osl_optimizer.step()
        self.encoder_optimizer.step()

    def update_osl_traj(self, replay_buffer):
        """Gets very low (1e-5) very quickly!"""
        self.osl.train(True)

        obses, actions, obses_next, rewards = replay_buffer.sample_traj(self.batch_size, self.k)

        loss = 0
        r_loss = 0
        z_o = self.osl.encoder_online(obses[:, 0, :, :, :])

        for i in range(self.k):
            # encoded
            z_m = self.osl.encoder_momentum(obses_next[:, i, :, :, :]).detach()

            # transition model
            z_o = self.osl.transition(z_o, actions[:, i])

            # reward prediction
            r_hat = self.osl.Wr(z_o)

            # # projections
            z_bar_o = self.osl.proj_online(z_o)
            z_bar_m = self.osl.proj_momentum(z_m).detach()

            # prediction
            z_hat_o = self.osl.predict(z_bar_o)

            # reward_pred
            # r_hat = self.osl.Wr(self.osl.encoder_online(obses[:, 0, :, :, :]), actions[:, i])
            # loss
            loss += self.byol_loss(z_hat_o, z_bar_m).mean()
            r_loss += F.mse_loss(r_hat, rewards[:, i])
            # loss += self.spr_loss(z_hat_o, z_bar_m)
            # r_loss += F.mse_loss(r_hat, rewards[:, i])

        # if np.random.rand() < 0.05:
        #     print(f'L: {loss.item()}, R: {r_loss.item()}')

        self.osl_loss_hist.append(loss.item())
        self.r_loss_hist.append(r_loss.item())

        combined_loss = loss + r_loss # * self.k # + (r_loss / self.k) #/ self.k

        self.osl_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        combined_loss.backward()

        # clip_grad_norm_(self.osl.parameters(), 10)
        # clip_grad_norm_(self.critic.encoder.parameters(), 10)

        self.osl_optimizer.step()
        self.encoder_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, logger, step)
        #
        if step % self.osl_update_frequency == 0:
            # for _ in range(2):
            # self.update_osl(obs, action, next_obs)
            # for _ in range(3):
            self.update_osl_traj(replay_buffer)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic.Q1, self.critic_target.Q1,
                                     0.01)
            utils.soft_update_params(self.critic.Q2, self.critic_target.Q2,
                                     0.01)
            utils.soft_update_params(self.osl.proj_online, self.osl.proj_momentum,
                                     0.05)
            utils.soft_update_params(self.osl.encoder_online, self.osl.encoder_momentum,
                                     0.05)

        # self.update_hidden(obs, next_obs)

    def pretrain(self, replay_buffer, step):
        # obs, action, reward, next_obs, not_done, obs_copy, next_obs_copy = replay_buffer.sample(self.batch_size)

        # self.update_osl(obs, action, next_obs)
        self.update_osl_traj(replay_buffer)

        # z = torch.FloatTensor(self.batch_size, self.critic.encoder.feature_dim).uniform_(0.8, 1.2).to(self.device)
        # z_two = torch.FloatTensor(self.batch_size, self.critic.encoder.feature_dim).uniform_(0.8, 1.2).to(self.device)
        #
        # self.update_osl(obs, action, next_obs, obs_copy, reward, z)
        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.osl.proj_online, self.osl.proj_momentum,
                                     0.05)
            utils.soft_update_params(self.osl.encoder_online, self.osl.encoder_momentum,
                                     0.05)
            # utils.soft_update_params(self.osl.online, self.osl.target, 0.05)

    def save(self, dir, extras):
        torch.save(
            self.actor.state_dict(), dir + extras + '_actor.pt'
        )

        torch.save(
            self.critic.state_dict(), dir + extras + '_critic.pt'
        )

        torch.save(
            self.osl.state_dict(), dir + extras + '_osl.pt'
        )

    def load(self, dir, extras):
        self.actor.load_state_dict(
            torch.load(dir + extras + '_actor.pt')
        )

        self.critic.load_state_dict(
            torch.load(dir + extras + '_critic.pt')
        )

        self.osl.load_state_dict(
            torch.load(dir + extras + '_osl.pt')
        )

def byol_loss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def spr_loss(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    l = 0
    for i in range(x.shape[0]):
        l += torch.dot(x[i], y[i].T)
    return -l

def spr_loss(x, y):
    x = F.normalize(x, dim=-1, p=2, eps=1e-3)
    y = F.normalize(y, dim=-1, p=2, eps=1e-3)
    loss = F.mse_loss(x, y, reduction='none').sum(dim=-1).mean(0)
    return loss