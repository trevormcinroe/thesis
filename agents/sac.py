"""
Official repo for SAC w/ discrete actions:
	https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/

Paper:
	https://arxiv.org/pdf/1910.07207.pdf
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from models.decoders import SACAEDecoder
from models.encoders import SACAEEncoder, SACAEEncoderDiscrete
from utils import weight_init, gaussian_logprob, squash, soft_update_params

"""SOFT ACTOR-CRITIC"""


################################ STATES ######################################
class SACActorStates(nn.Module):
	"""state_shape, num_layers, num_hidden, action_shape, is_discrete, device"""

	def __init__(self, state_shape, num_layers, num_hidden, action_shape, is_discrete=False, device='cpu'):
		super().__init__()

		self.is_discrete = is_discrete
		self.fc_trunk = nn.ModuleList([nn.Linear(state_shape, num_hidden), nn.ReLU()])
		for _ in range(num_layers - 1):
			self.fc_trunk.append(nn.Linear(num_hidden, num_hidden))
			self.fc_trunk.append(nn.ReLU())

		if is_discrete:
			self.d = nn.Linear(num_hidden, action_shape)
		else:
			pass

		self.apply(weight_init)
		self.to(device)

	def forward(self, s):
		for layer in self.fc_trunk:
			s = layer(s)

		if self.is_discrete:
			probs = F.softmax(self.d(s), dim=-1)
			return probs

		else:
			pass


# print(SACActorStates(9, 3, 256, 7, is_discrete=True)(torch.rand(1, 9)))

class SACVNetworkStates(nn.Module):
	def __init__(self, state_shape, num_layers, num_hidden, action_shape, device='cpu'):
		super().__init__()

		self.fc_trunk = nn.ModuleList([nn.Linear(state_shape, num_hidden), nn.ReLU()])
		for _ in range(num_layers - 1):
			self.fc_trunk.append(nn.Linear(num_hidden, num_hidden))
			self.fc_trunk.append(nn.ReLU())

		self.d = nn.Linear(num_hidden, action_shape)

		self.apply(weight_init)
		self.to(device)

	def forward(self, s):
		for layer in self.fc_trunk:
			s = layer(s)

		return self.d(s)


class SACCriticStates(nn.Module):
	def __init__(self, state_shape, num_layers, num_hidden, action_shape, device='cpu'):
		super().__init__()

		self.Q1 = SACVNetworkStates(state_shape, num_layers, num_hidden, action_shape, device)
		self.Q2 = SACVNetworkStates(state_shape, num_layers, num_hidden, action_shape, device)

	def forward(self, x):
		return self.Q1(x), self.Q2(x)


class SACAgentStates:
	def __init__(self, state_shape, num_layers, num_hidden, action_shape, actor_lr, actor_beta,
				 critic_lr, critic_beta, alpha_lr, alpha_beta, batch_size, critic_target_update_freq,
				 actor_update_freq, critic_tau, critic_update_freq, init_temperature=0.01, gamma=0.99,
				 is_discrete=False,
				 device='cpu', clip_grad=None):

		self.clip_grad = clip_grad
		self.device = device
		self.is_discrete = is_discrete
		self.gamma = gamma
		self.batch_size = batch_size
		self.critic_target_update_freq = critic_target_update_freq
		self.actor_update_freq = actor_update_freq
		self.critic_update_freq = critic_update_freq
		self.critic_tau = critic_tau

		self.actor = SACActorStates(state_shape, num_layers, num_hidden, action_shape, is_discrete, device)
		self.critic = SACCriticStates(state_shape, num_layers, num_hidden, action_shape, device)
		self.critic_target = SACCriticStates(state_shape, num_layers, num_hidden, action_shape, device)

		# copying weights from active critic to target critic
		self.critic_target.load_state_dict(self.critic.state_dict())

		# entropy term
		self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
		# self.log_alpha.requires_grad = True
		# self.log_alpha.to(device)

		# entropy target
		# self.target_entropy = -np.prod(action_shape)
		# Got this from the official GitHub of Discrete SAC
		self.target_entropy = -np.log((1.0 / action_shape)) * 0.98

		# optimizers
		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
		)

		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
		)

		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
		)

		# for reporting
		self.actor_losses = []
		self.critic_losses = []
		self.qs = []

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def greedy_action(self, s):
		"""Returns action with highest probability"""
		if self.is_discrete:
			action_probs = self.actor(s)
			return torch.argmax(action_probs).cpu().item()

		else:
			pass

	def sample_action(self, s, inspect=False):
		"""

		Args:
			s:

		Returns:
			sampled actions, log probs
		"""
		# This was originally torch.no_grad()... why?
		# with torch.no_grad():
		# A: perhaps there is some strange forward-pass accumulation that is causing
		# instabilities in the training process. (I hope?)
		# with torch.no_grad():
		action_probs = self.actor(s)
		action_dist = Categorical(action_probs)
		action = action_dist.sample()

		if not inspect:
			return action.cpu().item()
		else:
			return action.cpu().item(), action_probs

	def update(self, replay_buffer, step, report=False, encoder=None):

		s, a, r, s_, t = replay_buffer.sample(self.batch_size)

		# when using encoder, need to .detach() so we don't go through the compute graph twice
		if encoder is not None:
			s = encoder.encode_state(s.to(self.device)).detach()
			s_ = encoder.encode_state(s_.to(self.device)).detach()

		# (1) Critic
		# .unsqueeze(1): [1, 2, 3] --> [[1], [2], [3]]
		if step % self.critic_update_freq == 0:
			self.update_critic(
				s.to(self.device),
				a.to(self.device).unsqueeze(1),
				r.to(self.device).unsqueeze(1),
				s_.to(self.device),
				t.to(self.device).unsqueeze(1),
				report
			)

		# (2) Policy (3) Entropy
		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(s.to(self.device), report)

		# Soft update
		if step % self.critic_target_update_freq == 0:
			soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
			soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)

	def update_critic(self, s, a, r, s_, t, report):
		with torch.no_grad():
			action_probs = self.actor(s_)
			action_dist = Categorical(action_probs)
			action = action_dist.sample()

			# now need to get log_probs
			z = action_probs == 0
			z = z.float() * 1e-8
			log_prob = torch.log(action_probs + z)
			# log_prob = action_dist.log_prob(action)

			target_Q1, target_Q2 = self.critic_target(s_)

			# Pretty sure the .reshape(-1, 1) is necessary! Row vec -> col vec
			# target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob.reshape(-1, 1)
			# target_Q = r + (t * self.gamma * target_V)

			target_V = action_probs * (torch.min(target_Q1, target_Q2) - self.alpha * log_prob)
			target_V = target_V.sum(dim=1).unsqueeze(-1)
			target_Q = r + (t * self.gamma * target_V)

			if report:
				self.qs.append(torch.min(target_Q1, target_Q2).mean().item())

		current_Q1, current_Q2 = self.critic(s)
		current_Q1 = current_Q1.gather(1, a.long())
		current_Q2 = current_Q2.gather(1, a.long())

		# JQ = 𝔼_{st,at}~D[0.5(Q(st,at) - r(st,at) + γ(𝔼_{s_{t+1}~p}[V(s_{t+1})]))^2]
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()

		if self.clip_grad is not None:
			clip_grad_norm_(self.critic.parameters(), self.clip_grad)

		self.critic_optimizer.step()

		if report:
			self.critic_losses.append(critic_loss.item())

	def update_actor_and_alpha(self, s, report):
		"""NOTE: torch.no_grad() is unnecessary as the loss never gets propagated"""
		action_probs = self.actor(s)
		action_dist = Categorical(action_probs)
		action = action_dist.sample()

		# now need to get log_probs
		z = action_probs == 0
		z = z.float() * 1e-8
		log_prob = torch.log(action_probs + z)
		# log_prob = action_dist.log_prob(action)

		Q1, Q2 = self.critic(s)
		Q = torch.min(Q1, Q2)

		# TODO: detach?
		# Official implementation does NOT .detach() self.alpha here
		inside_term = (self.alpha * log_prob - Q)
		actor_loss = (action_probs * inside_term).sum(dim=1).mean()

		# alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

		# The official implementation uses log_alpha here....
		# But uses log_alpha.exp() everywhere else
		"""alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()"""

		# log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
		log_prob = torch.sum(log_prob * action_probs, dim=1)
		alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

		# Official implementation defers backpropf for actor AFTER entropy loss calculation.
		# Is there some weird accumulation/clearance issue here with PyTorch?
		self.actor_optimizer.zero_grad()
		actor_loss.backward()

		if self.clip_grad is not None:
			clip_grad_norm_(self.actor.parameters(), self.clip_grad)

		self.actor_optimizer.step()

		self.log_alpha_optimizer.zero_grad()
		alpha_loss.backward()

		# The official SAC-discrete repo does not clip the grad for alpha....
		# However, there is an issue with huge roller-coaster results with SAC...
		# if self.clip_grad is not None:
		# 	clip_grad_norm_([self.log_alpha])

		self.log_alpha_optimizer.step()

		if report:
			self.actor_losses.append(actor_loss.item())

	def clear_losses(self):
		self.actor_losses = []
		self.critic_losses = []
		self.qs = []

	def save(self, model_dir, step):
		torch.save(
			self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
		)
		torch.save(
			self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
		)

	def load(self, model_dir, step):
		self.actor.load_state_dict(
			torch.load('%s/actor_%s.pt' % (model_dir, step))
		)
		self.critic.load_state_dict(
			torch.load('%s/critic_%s.pt' % (model_dir, step))
		)


# agent = SACAgentStates(
# 	state_shape=9,
# 	num_layers=2,
# 	num_hidden=256,
# 	action_shape=7,
# 	actor_lr=3e-4,
# 	actor_beta=0.9,
# 	critic_lr=3e-4,
# 	critic_beta=0.9,
# 	alpha_lr=3e-4,
# 	alpha_beta=0.9,
# 	batch_size=256,
# 	critic_target_update_freq=1,
# 	actor_update_freq=2,  # critic should have a higher learning frequency than the actor
# 	critic_tau=0.005,
# 	is_discrete=True,
# 	device='cuda:0'
# )

################################ IMAGES ######################################
class SACActorImages(nn.Module):
	def __init__(self, state_cc, num_filters, num_convs, action_shape, is_discrete=False, device='cpu',
				 sigma_min=1e-6, sigma_max=2):
		super().__init__()

		self.is_discrete = is_discrete
		self.sigma_min = sigma_min
		self.sigma_max = sigma_max

		self.conv_trunk = nn.ModuleList([nn.Conv2d(state_cc, num_filters, (7, 7), 3), nn.ReLU()])

		for _ in range(num_convs - 2):
			self.conv_trunk.append(
				nn.Conv2d(num_filters, num_filters, (5, 5), 2)
			)
			self.conv_trunk.append(nn.ReLU())

		self.conv_trunk.append(nn.Conv2d(num_filters, num_filters, (5, 5), 2))
		self.conv_trunk.append(nn.ReLU())

		if is_discrete:
			self.d = nn.Linear(512, action_shape)
		else:
			self.d = nn.Linear(512, action_shape * 2)

		self.apply(weight_init)
		self.to(device)

	def forward(self, x):
		"""
		If self.is_discrete:
			returns a torch.distributions.Categorical(probs produced by softmax())

		else:
			returns a torch.distributions.Normal
		Args:
			x:

		Returns:
			probs, distribution class
		"""
		for conv in self.conv_trunk:
			x = conv(x)

		x = x.view(x.size(0), -1)

		if self.is_discrete:
			probs = F.softmax(self.d(x), dim=-1)
			return probs

		else:
			mus, sigmas = self.d(x).chunk(2, dim=-1)
			log_std = torch.clamp(sigmas, self.sigma_min, self.sigma_max)
			return mus, log_std


# def sample_action(self, s):
# 	""""""
# 	if self.is_discrete:
# 		probs = self.forward(s)
# 		action_dist = Categorical(probs)
# 		action = action_dist.sample()
# 		log_prob = action_dist.log_prob(action)
# 		return action.item(), log_prob

# 	else:
# 		pass
#
# def greedy_action(self, s):
# 	if self.is_discrete:
# 		probs = self.forward(s)
# 		return torch.argmax(probs).item()
#
# 	else:
# 		pass


# print(SACActorImages(state_cc=12, num_filters=32, num_convs=3, action_shape=4, is_discrete=True))
# .sample_action(
# 	torch.rand(1, 12, 84, 84)))


class SACVNetworkImages(nn.Module):
	def __init__(self, state_cc, num_filters, num_convs, action_shape, device='cpu'):
		super().__init__()

		self.conv_trunk = nn.ModuleList([nn.Conv2d(state_cc, num_filters, (7, 7), 3), nn.ReLU()])

		for _ in range(num_convs - 2):
			self.conv_trunk.append(
				nn.Conv2d(num_filters, num_filters, (5, 5), 2)
			)
			self.conv_trunk.append(nn.ReLU())

		self.conv_trunk.append(nn.Conv2d(num_filters, num_filters, (5, 5), 2))
		self.conv_trunk.append(nn.ReLU())

		self.d = nn.Linear(512, action_shape)

		self.apply(weight_init)
		self.to(device)

	def forward(self, x):
		for conv in self.conv_trunk:
			x = conv(x)

		x = x.view(x.size(0), -1)
		return self.d(x)


# print(SACVNetworkImages(state_cc=12, num_filters=32, num_convs=3))


class SACVCriticImages(nn.Module):
	""""""

	def __init__(self, state_cc, num_filters, num_convs, action_shape, device='cpu'):
		super().__init__()

		self.Q1 = SACVNetworkImages(state_cc, num_filters, num_convs, action_shape, device)
		self.Q2 = SACVNetworkImages(state_cc, num_filters, num_convs, action_shape, device)

	def forward(self, x):
		return self.Q1(x), self.Q2(x)


class SACAgentImages:
	def __init__(self, state_cc, num_filters, num_convs, action_shape, actor_lr, actor_beta,
				 critic_lr, critic_beta, alpha_lr, alpha_beta, batch_size, critic_target_update_freq,
				 actor_update_freq, critic_tau, critic_update_freq, init_temperature=0.01, gamma=0.99,
				 is_discrete=False, device='cpu', clip_grad=False):

		self.clip_grad = clip_grad
		self.device = device
		self.is_discrete = is_discrete
		self.gamma = gamma
		self.batch_size = batch_size
		self.critic_target_update_freq = critic_target_update_freq
		self.actor_update_freq = actor_update_freq
		self.critic_update_freq = critic_update_freq
		self.critic_tau = critic_tau

		self.actor = SACActorImages(state_cc, num_filters, num_convs, action_shape, is_discrete, device)
		self.critic = SACVCriticImages(state_cc, num_filters, num_convs, action_shape, device)
		self.critic_target = SACVCriticImages(state_cc, num_filters, num_convs, action_shape, device)

		# copying weights from active critic to target critic
		self.critic_target.load_state_dict(self.critic.state_dict())

		# entropy term
		self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
		# self.log_alpha.requires_grad = True
		# self.log_alpha.to(device)

		# entropy target
		# self.target_entropy = -np.prod(action_shape)
		# Got this from the official GitHub of Discrete SAC
		self.target_entropy = -np.log((1.0 / action_shape)) * 0.98

		# optimizers
		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
		)

		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
		)

		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
		)

		# for reporting
		self.actor_losses = []
		self.critic_losses = []
		self.qs = []

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def greedy_action(self, s):
		"""Returns action with highest probability"""
		if self.is_discrete:
			action_probs = self.actor(s)
			return torch.argmax(action_probs).cpu().item()

		else:
			pass

	def sample_action(self, s):
		"""

		Args:
			s:

		Returns:
			sampled actions, log probs
		"""
		# with torch.no_grad():
		action_probs = self.actor(s)
		action_dist = Categorical(action_probs)
		action = action_dist.sample()
		return action.cpu().item()

	def update(self, replay_buffer, step, report=False):

		s, a, r, s_, t = replay_buffer.sample(self.batch_size)

		# (1) Critic
		# .unsqueeze(1): [1, 2, 3] --> [[1], [2], [3]]
		if step % self.critic_update_freq == 0:
			self.update_critic(s.to(self.device),
							   a.to(self.device).unsqueeze(1),
							   r.to(self.device).unsqueeze(1),
							   s_.to(self.device),
							   t.to(self.device).unsqueeze(1),
							   report)

		# (2) Policy (3) Entropy
		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(s.to(self.device), report)

		# Soft update
		if step % self.critic_target_update_freq == 0:
			soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
			soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)

	def update_critic(self, s, a, r, s_, t, report):
		with torch.no_grad():
			action_probs = self.actor(s_)
			action_dist = Categorical(action_probs)
			action = action_dist.sample()

			# now need to get log_probs
			z = action_probs == 0
			z = z.float() * 1e-8
			log_prob = torch.log(action_probs + z)
			# log_prob = action_dist.log_prob(action)

			target_Q1, target_Q2 = self.critic_target(s_)

			# Pretty sure the .reshape(-1, 1) is necessary! Row vec -> col vec
			# target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob.reshape(-1, 1)
			# target_Q = r + (t * self.gamma * target_V)

			target_V = action_probs * (torch.min(target_Q1, target_Q2) - self.alpha * log_prob)
			target_V = target_V.sum(dim=1).unsqueeze(-1)
			target_Q = r + (t * self.gamma * target_V)

			if report:
				self.qs.append(torch.min(target_Q1, target_Q2).mean().item())

		current_Q1, current_Q2 = self.critic(s)
		current_Q1 = current_Q1.gather(1, a.long())
		current_Q2 = current_Q2.gather(1, a.long())

		# JQ = 𝔼_{st,at}~D[0.5(Q(st,at) - r(st,at) + γ(𝔼_{s_{t+1}~p}[V(s_{t+1})]))^2]
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()

		if self.clip_grad is not None:
			clip_grad_norm_(self.critic.parameters(), self.clip_grad)

		self.critic_optimizer.step()

		if report:
			self.critic_losses.append(critic_loss.item())

	def update_actor_and_alpha(self, s, report):
		"""NOTE: torch.no_grad() is unnecessary as the loss never gets propagated"""
		action_probs = self.actor(s)
		action_dist = Categorical(action_probs)
		action = action_dist.sample()

		# now need to get log_probs
		z = action_probs == 0
		z = z.float() * 1e-8
		log_prob = torch.log(action_probs + z)
		# log_prob = action_dist.log_prob(action)

		Q1, Q2 = self.critic(s)
		Q = torch.min(Q1, Q2)

		# TODO: detach?

		inside_term = (self.alpha * log_prob - Q)
		actor_loss = (action_probs * inside_term).sum(dim=1).mean()

		log_prob = torch.sum(log_prob * action_probs, dim=1)
		alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()

		if self.clip_grad is not None:
			clip_grad_norm_(self.actor.parameters(), self.clip_grad)

		self.actor_optimizer.step()

		self.log_alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.log_alpha_optimizer.step()

		if report:
			self.actor_losses.append(actor_loss.item())

	# alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

	# The official implementation uses log_alpha here....
	# But uses log_alpha.exp() everywhere else
	# alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
	# self.log_alpha_optimizer.zero_grad()
	# alpha_loss.backward()
	# self.log_alpha_optimizer.step()

	def clear_losses(self):
		self.actor_losses = []
		self.critic_losses = []
		self.qs = []

	def save(self, model_dir, step):
		torch.save(
			self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
		)
		torch.save(
			self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
		)

	def load(self, model_dir, step):
		self.actor.load_state_dict(
			torch.load('%s/actor_%s.pt' % (model_dir, step))
		)
		self.critic.load_state_dict(
			torch.load('%s/critic_%s.pt' % (model_dir, step))
		)


"""SAC-AE Discrete"""


class SACAEDiscreteActor(nn.Module):
	def __init__(self, state_cc, num_filters, num_layers, feature_dim,
				 action_shape, hidden_dim, device='cpu'):
		super().__init__()

		self.encoder = SACAEEncoderDiscrete(state_cc, feature_dim, num_filters)

		self.trunk = nn.ModuleList([
			nn.Linear(feature_dim, hidden_dim),
			nn.ReLU(),
		])

		for _ in range(num_layers - 1):
			self.trunk.append(nn.Linear(hidden_dim, hidden_dim))
			self.trunk.append(nn.ReLU())

		self.d = nn.Linear(hidden_dim, action_shape)

		self.apply(weight_init)
		self.to(device)

	def forward(self, x, detach_encoder=False):
		z = self.encoder(x, detach_encoder)

		for layer in self.trunk:
			z = layer(z)

		z = self.d(z)

		probs = F.softmax(z, dim=-1)
		return probs


class SACAEQNetwork(nn.Module):
	def __init__(self, z_dim, num_layers, num_hidden, action_shape, device='cpu'):
		super().__init__()

		self.trunk = nn.ModuleList([nn.Linear(z_dim, num_hidden), nn.ReLU()])
		for _ in range(num_layers - 1):
			self.trunk.append(nn.Linear(num_hidden, num_hidden))
			self.trunk.append(nn.ReLU())

		self.d = nn.Linear(num_hidden, action_shape)

		self.to(device)

	def forward(self, x):
		for layer in self.trunk:
			x = layer(x)

		return self.d(x)


class SACAEDiscreteCritic(nn.Module):
	def __init__(self, z_dim, num_layers, num_hidden, action_shape,
				 state_cc, num_filters, device='cpu'):
		super().__init__()

		self.encoder = SACAEEncoderDiscrete(state_cc, z_dim, num_filters, device)

		self.Q1 = SACAEQNetwork(z_dim, num_layers, num_hidden,
								action_shape, device)

		self.Q2 = SACAEQNetwork(z_dim, num_layers, num_hidden,
								action_shape, device)

	def forward(self, x, detach_encoder=False):
		x = self.encoder(x, detach_encoder)
		return self.Q1(x), self.Q2(x)


class SACAEAgentDiscrete:
	def __init__(
			self,
			state_cc,
			action_shape,
			num_layers,
			num_hidden,
			gamma=0.99,
			alpha_lr=1e-3,
			alpha_beta=0.9,
			actor_lr=1e-3,
			actor_beta=0.9,
			critic_update_freq=2,
			actor_update_freq=2,
			critic_lr=1e-3,
			critic_beta=0.9,
			critic_tau=0.005,
			critic_target_update_freq=2,
			z_dim=50,
			encoder_lr=1e-3,
			encoder_tau=0.005,
			decoder_lr=1e-3,
			decoder_update_freq=1,
			decoder_latent_lambda=0.0,
			decoder_weight_lambda=0.0,
			num_filters=32,
			device='cpu',
			clip_grad=5.0,
			batch_size=32
	):
		self.clip_grad = clip_grad
		self.device = device
		self.gamma = gamma
		self.decoder_latent_lambda = decoder_latent_lambda
		self.actor_update_freq = actor_update_freq
		self.critic_tau = critic_tau
		self.critic_target_update_freq = critic_target_update_freq
		self.encoder_tau = encoder_tau
		self.decoder_update_freq = decoder_update_freq
		self.batch_size = batch_size
		self.critic_update_freq = critic_update_freq

		self.actor = SACAEDiscreteActor(
			state_cc, num_filters, num_layers, z_dim, action_shape,
			num_hidden, device
		)

		self.critic = SACAEDiscreteCritic(
			z_dim, num_layers, num_hidden, action_shape, state_cc,
			num_filters, device
		)

		self.critic_target = SACAEDiscreteCritic(
			z_dim, num_layers, num_hidden, action_shape, state_cc,
			num_filters, device
		)

		# copying weights from active critic to target critic
		self.critic_target.load_state_dict(self.critic.state_dict())

		# Need to tie the two disparate encoders
		self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

		# entropy term
		self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
		# self.log_alpha.requires_grad = True
		# self.log_alpha.to(device)

		# entropy target
		# self.target_entropy = -np.prod(action_shape)
		# Got this from the official GitHub of Discrete SAC
		self.target_entropy = -np.log((1.0 / action_shape)) * 0.98

		self.decoder = SACAEDecoder(state_cc, z_dim, num_filters, device)
		self.decoder.apply(weight_init)

		# Setting up the optimizers
		# TODO: check if these are the settings from the paper
		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
		)

		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
		)

		# Entropy term is a scalar and torch's optimizers need an iterable, thus [entropy]
		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
		)

		# This is where the L2 norm on the decoder's weights is specified
		self.decoder_optimizer = torch.optim.Adam(
			self.decoder.parameters(), lr=decoder_lr, weight_decay=decoder_weight_lambda
		)

		self.encoder_optimizer = torch.optim.Adam(
			self.critic.encoder.parameters(), lr=encoder_lr
		)

		# for reporting
		self.actor_losses = []
		self.critic_losses = []
		self.qs = []

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def sample_action(self, s):
		action_probs = self.actor(s)
		action_dist = Categorical(action_probs)
		action = action_dist.sample()

		return action.cpu().item()

	def update(self, replay_buffer, step, report=False):
		s, a, r, s_, t = replay_buffer.sample(self.batch_size)

		if step % self.critic_update_freq == 0:
			self.update_critic(
				s.to(self.device),
				a.to(self.device).unsqueeze(1),
				r.to(self.device).unsqueeze(1),
				s_.to(self.device),
				t.to(self.device).unsqueeze(1),
				report
			)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(s.to(self.device), report)

		if step % self.critic_target_update_freq == 0:
			soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
			soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
			soft_update_params(self.critic.encoder, self.critic_target.encoder, self.critic_tau)

		if step % self.decoder_update_freq == 0:
			self.update_decoder(s.to(self.device), s.to(self.device))

	def update_critic(self, s, a, r, s_, t, report):
		with torch.no_grad():
			action_probs = self.actor(s_)
			action_dist = Categorical(action_probs)
			action = action_dist.sample()

			# now need to get log_probs
			z = action_probs == 0
			z = z.float() * 1e-8
			log_prob = torch.log(action_probs + z)
			# log_prob = action_dist.log_prob(action)

			target_Q1, target_Q2 = self.critic_target(s_)

			# Pretty sure the .reshape(-1, 1) is necessary! Row vec -> col vec
			# target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob.reshape(-1, 1)
			# target_Q = r + (t * self.gamma * target_V)

			target_V = action_probs * (torch.min(target_Q1, target_Q2) - self.alpha * log_prob)
			target_V = target_V.sum(dim=1).unsqueeze(-1)
			target_Q = r + (t * self.gamma * target_V)

			if report:
				self.qs.append(torch.min(target_Q1, target_Q2).mean().item())

		current_Q1, current_Q2 = self.critic(s)
		current_Q1 = current_Q1.gather(1, a.long())
		current_Q2 = current_Q2.gather(1, a.long())

		# JQ = 𝔼_{st,at}~D[0.5(Q(st,at) - r(st,at) + γ(𝔼_{s_{t+1}~p}[V(s_{t+1})]))^2]
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()

		if self.clip_grad is not None:
			clip_grad_norm_(self.critic.parameters(), self.clip_grad)

		self.critic_optimizer.step()

		if report:
			self.critic_losses.append(critic_loss.item())

	def update_actor_and_alpha(self, s, report):
		# The paper explains that we detach the encoder here because we DO NOT want to let the gradient
		# backpropagate into the encoder via the actor!
		action_probs = self.actor(s, detach_encoder=True)
		action_dist = Categorical(action_probs)
		action = action_dist.sample()

		# now need to get log_probs
		z = action_probs == 0
		z = z.float() * 1e-8
		log_prob = torch.log(action_probs + z)
		# log_prob = action_dist.log_prob(action)

		Q1, Q2 = self.critic(s, detach_encoder=True)
		Q = torch.min(Q1, Q2)

		# TODO: detach?
		# Official implementation does NOT .detach() self.alpha here
		inside_term = (self.alpha * log_prob - Q)
		actor_loss = (action_probs * inside_term).sum(dim=1).mean()

		# alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

		# The official implementation uses log_alpha here....
		# But uses log_alpha.exp() everywhere else
		"""alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()"""

		# log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
		log_prob = torch.sum(log_prob * action_probs, dim=1)
		alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

		# Official implementation defers backpropf for actor AFTER entropy loss calculation.
		# Is there some weird accumulation/clearance issue here with PyTorch?
		self.actor_optimizer.zero_grad()
		actor_loss.backward()

		if self.clip_grad is not None:
			clip_grad_norm_(self.actor.parameters(), self.clip_grad)

		self.actor_optimizer.step()

		self.log_alpha_optimizer.zero_grad()
		alpha_loss.backward()

		# The official SAC-discrete repo does not clip the grad for alpha....
		# However, there is an issue with huge roller-coaster results with SAC...
		# if self.clip_grad is not None:
		# 	clip_grad_norm_([self.log_alpha])

		self.log_alpha_optimizer.step()

		if report:
			self.actor_losses.append(actor_loss.item())

	def update_decoder(self, s, s_target):
		h = self.critic.encoder(s)

		# At this point, the paper's codebase norms the targets to be [-0.5, 0.5] and references the
		# GLOW paper by Kingma. The repo for this paper has a preprocess() function that does this but I cannot
		# seem to find a reference in the paper itself...?
		# For now, we will skip.
		rec_obs = self.decoder(h)
		rec_loss = F.mse_loss(s_target, rec_obs)

		# add L2 penalty on latent representation
		# see https://arxiv.org/pdf/1903.12436.pdf
		latent_loss = (0.5 * h.pow(2).sum(1)).mean()

		# by default self.decoder_latent_lambda = 0.0 ??
		loss = rec_loss + self.decoder_latent_lambda * latent_loss

		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()
		loss.backward()
		self.encoder_optimizer.step()
		self.decoder_optimizer.step()

	def clear_losses(self):
		self.actor_losses = []
		self.critic_losses = []
		self.qs = []

	def save(self, model_dir, step):
		torch.save(
			self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
		)
		torch.save(
			self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
		)

	def load(self, model_dir, step):
		self.actor.load_state_dict(
			torch.load('%s/actor_%s.pt' % (model_dir, step))
		)
		self.critic.load_state_dict(
			torch.load('%s/critic_%s.pt' % (model_dir, step))
		)


"""SAC-AE"""


class SACAEContinuousActor(nn.Module):
	def __init__(self, state_cc, encoder_feature_dim, num_filters,
				 hidden_dim, action_shape, log_std_min, log_std_max, device='cpu'):
		super().__init__()

		self.encoder = SACAEEncoder(state_cc, encoder_feature_dim, num_filters, device)

		self.log_std_min = log_std_min
		self.log_std_max = log_std_max

		self.trunk = nn.Sequential(
			nn.Linear(self.encoder.feature_dim, hidden_dim),
			nn.RelU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.RelU(),
			nn.Linear(hidden_dim, 2 * action_shape[0])
		)

		self.apply(weight_init)
		self.to(device)

	def forward(self, s, compute_pi=True, compute_log_pi=True, detach_encoder=False):
		z = self.encoder(s, detach_encoder)

		mu, log_std = self.trunk(z).chunk(2, dim=-1)

		# constrain log_std inside [log_std_min, log_std_max]
		log_std = torch.tanh(log_std)
		log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

		if compute_pi:
			std = log_std.exp()
			noise = torch.randn_like(mu)
			pi = mu + noise * std
		else:
			pi = None
			entropy = None

		if compute_log_pi:
			log_pi = gaussian_logprob(noise, log_std)
		else:
			log_pi = None

		mu, pi, log_pi = squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std


class QFunction(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim, device='cpu'):
		super().__init__()

		self.trunk = nn.Sequential(
			nn.Linear(obs_dim + action_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1)
		)

		self.to(device)

	def forward(self, obs, action):
		assert obs.size(0) == action.size(0)

		obs_action = torch.cat([obs, action], dim=1)
		return self.trunk(obs_action)


class SACAECritic(nn.Module):
	def __init__(self, state_cc, encoder_feature_dim, num_filters,
				 action_shape, hidden_dim, device='cpu'):
		super().__init__()

		self.encoder = SACAEEncoder(state_cc, encoder_feature_dim, num_filters, device)

		self.Q1 = QFunction(
			self.encoder.feature_dim, action_shape[0], hidden_dim
		)
		self.Q2 = QFunction(
			self.encoder.feature_dim, action_shape[0], hidden_dim
		)

		self.apply(weight_init)
		self.to(device)

	def forward(self, s, a, detach_encoder=False):
		z = self.encoder(s, detach_encoder)

		q1 = self.Q1(s, a)
		q2 = self.Q2(s, a)

		return q1, q2


class SACAEAgent:
	def __init__(
			self,
			state_cc,
			action_shape,
			hidden_dim=256,
			discount=0.99,
			init_temperature=0.01,
			alpha_lr=1e-3,
			alpha_beta=0.9,
			actor_lr=1e-3,
			actor_beta=0.9,
			actor_log_std_min=-10,
			actor_log_std_max=2,
			actor_update_freq=2,
			critic_lr=1e-3,
			critic_beta=0.9,
			critic_tau=0.005,
			critic_target_update_freq=2,
			encoder_feature_dim=50,
			encoder_lr=1e-3,
			encoder_tau=0.005,
			decoder_lr=1e-3,
			decoder_update_freq=1,
			decoder_latent_lambda=0.0,
			decoder_weight_lambda=0.0,
			num_filters=32,
			device='cpu'
	):
		self.device = device
		self.discount = discount
		self.decoder_latent_lambda = decoder_latent_lambda
		self.actor_update_freq = actor_update_freq
		self.critic_tau = critic_tau
		self.critic_target_update_freq = critic_target_update_freq
		self.encoder_tau = encoder_tau
		self.decoder_update_freq = decoder_update_freq

		self.actor = SACAEContinuousActor(
			state_cc, encoder_feature_dim, num_filters,
			actor_log_std_min, actor_log_std_max, hidden_dim,
			action_shape, device
		)

		self.critic = SACAECritic(
			state_cc, encoder_feature_dim, num_filters,
			action_shape, hidden_dim, device
		)

		self.critic_target = SACAECritic(
			state_cc, encoder_feature_dim, num_filters,
			action_shape, hidden_dim, device
		)

		# Copying weights from critic to target
		self.critic_target.load_state_dict(self.critic.state_dict())

		# Need to tie the two disparate encoders
		self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

		# TODO: why this? In the paper, entropy term = 0.1 -- why log?
		self.log_alpha = torch.tensor(np.log(init_temperature))

		# Entropy term needs to be learning
		self.log_alpha.requires_grad = True

		# Section D of appendix https://arxiv.org/pdf/1812.05905.pdf (SAC PAPER)
		self.target_entropy = -np.prod(action_shape)

		self.decoder = SACAEDecoder(
			state_cc, encoder_feature_dim, num_filters, device
		)

		# I believe the weight init for the encoders is performed within the Actor/Critic classes themselves.
		# Is this the best/most-clear paradigm? Maybe not...
		self.decoder.apply(weight_init)

		# Setting up the optimizers
		# TODO: check if these are the settings from the paper
		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
		)

		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
		)

		# Entropy term is a scalar and torch's optimizers need an iterable, thus [entropy]
		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
		)

		# This is where the L2 norm on the decoder's weights is specified
		self.decoder_optimizer = torch.optim.Adam(
			self.decoder.parameters(), lr=decoder_lr, weight_decay=decoder_weight_lambda
		)

		self.encoder_optimizer = torch.optim.Adam(
			self.critic.encoder.parameters(), lr=encoder_lr
		)

		# TODO: is it necessary to explicitly set critic_target separately?
		self.train()
		self.critic_target.train()

	def train(self, training=True):
		"""Not sure if this is necessary..."""
		self.training = training
		self.actor.train(training)
		self.critic.train(training)
		if self.decoder is not None:
			self.decoder.train(training)

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def select_action(self, s):
		"""The paradigm from the currently existing experiments is to deal with axis-shifting OUTSIDE of the
			model classes.

			Returns a deterministic action via the mean of the distribution
		"""
		with torch.no_grad():
			mu, _, _, _ = self.actor(s.to(self.device), compute_pi=False, compute_log_pi=False)

			# TODO: determine if flattening is necessary??
			return mu.cpu().data.numpy().flatten()

	def sample_actions(self, s):
		"""Returns a probabilistic action via pi"""
		with torch.no_grad():
			_, pi, _, _ = self.actor(s.to(self.device), compute_log_pi=False)

			# TODO: determine if flattening is necessary??
			return pi.cpu().data.numpy().flatten()

	def update_critic(self, s, a, r, s_, not_done):
		"""

		Args:
			s:
			a:
			r:
			s_:
			not_done:
			L:
			step:

		Returns:

		"""
		# When updating critic, don't need gradient for actor nor the target critic networks!
		with torch.no_grad():
			# Next state!
			_, policy_action, log_pi, _ = self.actor(s_.to(self.device))
			target_Q1, target_Q2 = self.critic_target(s_.to(self.device), policy_action)

			# per SAC paper, selecting the minimum of the two Q-networks
			# TODO: why subtracting alpha * log_pi?
			target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi

			target_Q = r + (not_done * self.discount * target_V)

		# Current critic passthru
		current_Q1, current_Q2 = self.critic(s.to(self.device), a)

		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update_actor_and_alpha(self, s):
		"""

		Args:
			s:

		Returns:

		"""
		# The paper explains that we detach the encoder here because we DO NOT want to let the gradient
		# backpropagate into the encoder via the actor!
		_, pi, log_pi, log_std = self.actor(s.to(self.device), detach_encoder=True)
		actor_Q1, actor_Q2 = self.critic(s.to(self.device), pi, detach_encoder=True)

		# TODO: determine why no - alpha * log_pi until following step
		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

		# TODO: ???
		entropy = 0.5 * log_std.shape[1] * (1. + np.log(2 * np.pi)) + log_std.sum(dim=-1)

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.log_alpha_optimizer.zero_grad()

		# TODO: ???
		alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
		alpha_loss.backward()
		self.log_alpha_optimizer.step()

	def update_decoder(self, s, s_target):
		h = self.critic.encoder(s.to(self.device))

		# At this point, the paper's codebase norms the targets to be [-0.5, 0.5] and references the
		# GLOW paper by Kingma. The repo for this paper has a preprocess() function that does this but I cannot
		# seem to find a reference in the paper itself...?
		# For now, we will skip.
		rec_obs = self.decoder(h)
		rec_loss = F.mse_loss(s_target, rec_obs)

		# add L2 penalty on latent representation
		# see https://arxiv.org/pdf/1903.12436.pdf
		latent_loss = (0.5 * h.pow(2).sum(1)).mean()

		# by default self.decoder_latent_lambda = 0.0 ??
		loss = rec_loss + self.decoder_latent_lambda * latent_loss

		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()
		loss.backward()
		self.encoder_optimizer.step()
		self.decoder_optimizer.step()

	def update(self, replay_buffer, step):
		"""

		Args:
			replay_buffer:
			step:

		Returns:

		"""

		s, a, r, s_, t = replay_buffer.sample()

		self.update_critic(s, a, r, s_, t)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(s)

		if step % self.critic_target_update_freq == 0:
			soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)

			soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)

			soft_update_params(self.critic.encoder, self.critic_target.encoder, self.critic_tau)

		if step % self.decoder_update_freq == 0:
			self.update_decoder(s, s)

	def save(self, model_dir, step):
		torch.save(
			self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
		)
		torch.save(
			self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
		)
		if self.decoder is not None:
			torch.save(
				self.decoder.state_dict(),
				'%s/decoder_%s.pt' % (model_dir, step)
			)

	def load(self, model_dir, step):
		self.actor.load_state_dict(
			torch.load('%s/actor_%s.pt' % (model_dir, step))
		)
		self.critic.load_state_dict(
			torch.load('%s/critic_%s.pt' % (model_dir, step))
		)
		if self.decoder is not None:
			self.decoder.load_state_dict(
				torch.load('%s/decoder_%s.pt' % (model_dir, step))
			)


"""CONTINUOUS VERSIONS!"""


class SACContinuousActorImages(nn.Module):
	"""MLP actor network."""

	def __init__(self, state_cc, num_filters, num_convs, action_shape, device='cpu', sigma_min=1e-6, sigma_max=2):
		super().__init__()

		self.sigma_min = sigma_min
		self.sigma_max = sigma_max

		self.conv_trunk = nn.ModuleList([nn.Conv2d(state_cc, num_filters, (7, 7), 3), nn.ReLU()])

		for _ in range(num_convs - 2):
			self.conv_trunk.append(
				nn.Conv2d(num_filters, num_filters, (5, 5), 2)
			)
			self.conv_trunk.append(nn.ReLU())

		self.conv_trunk.append(nn.Conv2d(num_filters, num_filters, (5, 5), 2))
		self.conv_trunk.append(nn.ReLU())

		self.d1 = nn.Linear(512, 2 * action_shape)

		self.apply(weight_init)
		self.to(device)

	def forward(self, x, compute_pi=True, compute_log_pi=True):

		for conv in self.conv_trunk:
			x = conv(x)

		x = x.view(x.size(0), -1)

		mu, log_std = self.d1(x).chunk(2, dim=-1)

		# constrain log_std inside [log_std_min, log_std_max]
		log_std = torch.tanh(log_std)
		log_std = self.sigma_min + 0.5 * (
				self.sigma_max - self.sigma_min
		) * (log_std + 1)

		# self.outputs['mu'] = mu
		# self.outputs['std'] = log_std.exp()

		if compute_pi:
			std = log_std.exp()
			noise = torch.randn_like(mu)
			pi = mu + noise * std
		else:
			pi = None
			entropy = None

		if compute_log_pi:
			log_pi = gaussian_logprob(noise, log_std)
		else:
			log_pi = None

		mu, pi, log_pi = squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std


class SACContinuousAgentImages:
	def __init__(self, state_cc, num_filters, num_convs, action_shape, actor_lr, actor_beta,
				 critic_lr, critic_beta, alpha_lr, alpha_beta, batch_size, critic_target_update_freq,
				 actor_update_freq, critic_tau, critic_update_freq, init_temperature=0.01, gamma=0.99,
				 is_discrete=False, device='cpu', clip_grad=False):
		self.clip_grad = clip_grad
		self.device = device
		self.is_discrete = is_discrete
		self.gamma = gamma
		self.batch_size = batch_size
		self.critic_target_update_freq = critic_target_update_freq
		self.actor_update_freq = actor_update_freq
		self.critic_update_freq = critic_update_freq
		self.critic_tau = critic_tau

		self.actor = SACContinuousActorImages(state_cc, num_filters, num_convs, action_shape, device)
		self.critic = SACVCriticImages(state_cc, num_filters, num_convs, action_shape, device)
		self.critic_target = SACVCriticImages(state_cc, num_filters, num_convs, action_shape, device)

		# copying weights from active critic to target critic
		self.critic_target.load_state_dict(self.critic.state_dict())

		self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
		self.log_alpha.requires_grad = True
		# set target entropy to -|A|
		self.target_entropy = -np.prod(action_shape)

		# optimizers
		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
		)

		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
		)

		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
		)

		# for reporting
		self.actor_losses = []
		self.critic_losses = []
		self.qs = []

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def select_action(self, obs):
		"""Deterministic action-selection"""
		with torch.no_grad():
			# obs = torch.FloatTensor(obs).to(self.device)
			# obs = obs.unsqueeze(0)
			mu, _, _, _ = self.actor(
				obs, compute_pi=False, compute_log_pi=False
			)
			return mu.cpu().data.numpy().flatten()

	def sample_action(self, obs):
		with torch.no_grad():
			_, pi, _, _ = self.actor(obs, compute_log_pi=False)
			return pi.cpu().data.numpy().flatten()

	def update(self, replay_buffer, step, report=False):
		s, a, r, s_, t = replay_buffer.sample(self.batch_size)

		# (1) Critic
		# .unsqueeze(1): [1, 2, 3] --> [[1], [2], [3]]
		if step % self.critic_update_freq == 0:
			self.update_critic(s.to(self.device),
							   a.to(self.device).unsqueeze(1),
							   r.to(self.device).unsqueeze(1),
							   s_.to(self.device),
							   t.to(self.device).unsqueeze(1),
							   report)

		# (2) Policy (3) Entropy
		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(s.to(self.device), report)

		# Soft update
		if step % self.critic_target_update_freq == 0:
			soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
			soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)

	def update_critic(self, obs, action, reward, next_obs, not_done, report):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs)
			target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.gamma * target_V)

			if report:
				self.qs.append(torch.min(target_Q1, target_Q2).mean().item())

		# get current Q estimates
		current_Q1, current_Q2 = self.critic(obs)

		# JQ = 𝔼_{st,at}~D[0.5(Q(st,at) - r(st,at) + γ(𝔼_{s_{t+1}~p}[V(s_{t+1})]))^2]
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)


		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()

		if self.clip_grad is not None:
			clip_grad_norm_(self.critic.parameters(), self.clip_grad)

		self.critic_optimizer.step()

		if report:
			self.critic_losses.append(critic_loss.item())

	def update_actor_and_alpha(self, obs, report):
		_, pi, log_pi, log_std = self.actor(obs)
		actor_Q1, actor_Q2 = self.critic(obs)
		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

		# TODO: determine why this is here
		entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

		self.actor_optimizer.zero_grad()
		actor_loss.backward()

		if self.clip_grad is not None:
			clip_grad_norm_(self.actor.parameters(), self.clip_grad)

		self.actor_optimizer.step()

		self.log_alpha_optimizer.zero_grad()
		alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

		alpha_loss.backward()
		self.log_alpha_optimizer.step()

		if report:
			self.actor_losses.append(actor_loss.item())

	def clear_losses(self):
		self.actor_losses = []
		self.critic_losses = []
		self.qs = []

	def save(self, model_dir, step):
		torch.save(
			self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
		)
		torch.save(
			self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
		)
		if self.decoder is not None:
			torch.save(
				self.decoder.state_dict(),
				'%s/decoder_%s.pt' % (model_dir, step)
			)

	def load(self, model_dir, step):
		self.actor.load_state_dict(
			torch.load('%s/actor_%s.pt' % (model_dir, step))
		)
		self.critic.load_state_dict(
			torch.load('%s/critic_%s.pt' % (model_dir, step))
		)
		if self.decoder is not None:
			self.decoder.load_state_dict(
				torch.load('%s/decoder_%s.pt' % (model_dir, step))
			)

# def log(self, L, step, log_freq=LOG_FREQ):
#     if step % log_freq != 0:
#         return
#
#     for k, v in self.outputs.items():
#         L.log_histogram('train_actor/%s_hist' % k, v, step)
#
#     L.log_param('train_actor/fc1', self.trunk[0], step)
#     L.log_param('train_actor/fc2', self.trunk[2], step)
#     L.log_param('train_actor/fc3', self.trunk[4], step)
