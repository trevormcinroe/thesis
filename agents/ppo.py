from copy import deepcopy
import torch.nn.functional as F
from torch import nn
from torch import optim
import torch
from torch.distributions import Categorical
import sys


class PPOActorStates(nn.Module):
	def __init__(self, num_layers, num_hidden, state_shape, action_shape):
		super().__init__()

		self.trunk = nn.ModuleList([nn.Linear(state_shape, num_hidden), nn.ReLU()])
		for _ in range(num_layers - 2):
			self.trunk.append(nn.Linear(num_hidden, num_hidden))
			self.trunk.append(nn.ReLU())

		self.trunk.append(nn.Linear(num_hidden, action_shape))

	def forward(self, s):
		for layer in self.trunk:
			s = layer(s)

		s = F.softmax(s, dim=-1)
		return s


class PPOAgentStates:
	def __init__(self, num_layers, num_hidden, state_shape, action_shape, lr,
				 episodes_per_learning_round, learning_iterations_per_round, gamma,
				 clip_eps, experience_generator, clip_grad=None):
		self.policy_new = PPOActorStates(num_layers, num_hidden, state_shape, action_shape)
		self.policy_old = PPOActorStates(num_layers, num_hidden, state_shape, action_shape)
		self.policy_old.load_state_dict(deepcopy(self.policy_new.state_dict()))

		self.policy_new_optimizer = optim.Adam(self.policy_new.parameters(), lr=lr, eps=1e-4)

		self.episodes_per_learning_round = episodes_per_learning_round
		self.learning_iterations_per_round = learning_iterations_per_round
		self.gamma = gamma
		self.clip_eps = clip_eps
		self.clip_grad = clip_grad

		self.many_episode_states = []
		self.many_episode_actions = []
		self.many_episode_rewards = []

		self.experience_generator = experience_generator

	def step(self):
		"""Runs a single learning step for the agent"""
		# For now, let's skip the epsilon exploration...
		generated_output = self.experience_generator.play_n_episodes(
			self.episodes_per_learning_round, self.policy_new
		)

		collected_states = generated_output[0]
		collected_actions = generated_output[1]
		collected_rewards = generated_output[2]
		collected_completed = generated_output[3]
		global_steps = generated_output[4]

		self.many_episode_states = collected_states
		self.many_episode_actions = collected_actions
		self.many_episode_rewards = collected_rewards

		self.policy_learn()
		# another learning rate update step here...
		self.equalize_policies()

		return collected_completed, global_steps

	def policy_learn(self):
		all_discounted_returns = self.calculate_all_discounted_returns()

		for _ in range(self.learning_iterations_per_round):
			all_ratio_of_policy_probs = self.calculate_all_ratio_of_policy_probs()
			loss = self.calculate_loss([all_ratio_of_policy_probs], all_discounted_returns)
			self.take_policy_new_optimization_step(loss)


	def calculate_all_discounted_returns(self):
		"""Calculates the cumulative discounted return for each episode which we will
		then use in a learning iteration
		"""
		all_discounted_returns = []

		# looping through each collected episode
		for episode in range(len(self.many_episode_states)):
			# 0 in beginning to have 0 be for terminal state??
			# perhaps this is fine here. Most mentions of time limit being 0 are referencing value-based methods
			discounted_returns = [0]

			# looping through each step of the inner episode loop
			for idx in range(len(self.many_episode_states[episode])):
				# the indexing is negative because we are working backwards
				return_value = self.many_episode_rewards[episode][-(idx + 1)] \
							   + self.gamma * discounted_returns[-1]
				discounted_returns.append(return_value)

			# remove that beginning 0
			discounted_returns = discounted_returns[1:]

			# extending object with reversed (which makes it forward) rewards
			all_discounted_returns.extend(discounted_returns[::-1])
		return all_discounted_returns

	def calculate_all_ratio_of_policy_probs(self):
		"""For each action, calculate the ratio Pr(a|pi_new) / Pr(a|pi_old)"""
		# super unfornate syntax. unrolls list of lists into a single list
		all_states = [state.float() for states in self.many_episode_states for state in states]
		all_actions = [[action] for actions in self.many_episode_actions for action in actions]

		all_states = torch.stack([torch.Tensor(states).float() for states in all_states])
		all_actions = torch.stack([torch.Tensor(actions).float() for actions in all_actions])

		# flattens => tensor([1., ..., n])
		all_actions = all_actions.view(-1, len(all_states))

		new_pi_log_prob = self.calculate_log_probs(self.policy_new, all_states, all_actions)
		old_pi_log_prob = self.calculate_log_probs(self.policy_old, all_states, all_actions)
		ratio = torch.exp(new_pi_log_prob) / (torch.exp(old_pi_log_prob) + 1e-8)
		return ratio

	def calculate_log_probs(self, policy, states, actions):
		policy_output = policy(states)
		policy_dist = Categorical(policy_output)
		return policy_dist.log_prob(actions)

	def calculate_loss(self, ratios, all_discounted_returns):
		ratios = torch.squeeze(torch.stack(ratios))

		# computational stability granting (?)
		ratios = torch.clamp(input=ratios, min=-sys.maxsize, max=sys.maxsize)

		all_discounted_returns = torch.tensor(all_discounted_returns).to(ratios)

		loss_1 = all_discounted_returns * ratios
		loss_2 = all_discounted_returns * self.clamp_prob_ratio(ratios)
		loss = torch.min(loss_1, loss_2)
		return -torch.mean(loss)

	def clamp_prob_ratio(self, ratios):
		return torch.clamp(
			ratios, min=1.0 - self.clip_eps, max=1.0 + self.clip_eps
		)

	def take_policy_new_optimization_step(self, loss):
		self.policy_new_optimizer.zero_grad()
		loss.backward()
		if self.clip_grad is not None:
			torch.nn.utils.clip_grad_norm_(self.policy_new.parameters(), self.clip_grad)
		self.policy_new_optimizer.step()

	def equalize_policies(self):
		for old_param, new_param in zip(self.policy_old.parameters(), self.policy_new.parameters()):
			old_param.data.copy_(new_param.data)
