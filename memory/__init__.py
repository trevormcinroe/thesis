import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# TODO: how to norm rewards?


class ReplayMemory:

	def __init__(self, mem_size, s_shape, a_shape=0, norm=False):
		self.mem_size = mem_size
		if np.any([isinstance(s_shape, list), isinstance(s_shape, tuple)]):
			self.s = torch.zeros((mem_size, *s_shape))
			self.a = torch.zeros((mem_size, a_shape))
			self.r = torch.zeros(mem_size)
			self.s_ = torch.zeros((mem_size, *s_shape))
			self.t = torch.zeros(mem_size)

		else:
			self.s = torch.zeros((mem_size, s_shape))
			self.a = torch.zeros((mem_size, a_shape))
			self.r = torch.zeros(mem_size)
			self.s_ = torch.zeros((mem_size, s_shape))
			self.t = torch.zeros(mem_size)

		self.mem_cntr = 0

		self.norm = norm
		self.mean = None
		self.std = None

	def store(self, s, a, r, s_, t):
		idx = self.mem_cntr % self.mem_size

		self.s[idx] = s
		self.a[idx] = a
		self.r[idx] = r
		self.s_[idx] = s_
		self.t[idx] = 1 - t

		self.mem_cntr += 1

	def sample(self, batch_size):
		"""We should probably handle the max_size<batch_size elsewhere?"""
		max_size = min(self.mem_cntr, self.mem_size)
		idxs = np.random.choice(max_size, size=batch_size, replace=False)

		if self.norm:
			self.mean = torch.mean(self.r[:max_size])
			self.std = torch.std(self.r[:max_size])
			r = self.r[idxs]
			r = (r - self.mean) / (self.std + 1e-8)
			return self.s[idxs], self.a[idxs], r, self.s_[idxs], self.t[idxs]

		else:
			return self.s[idxs], self.a[idxs], self.r[idxs], self.s_[idxs], self.t[idxs]


class RolloutStorage:
	def __init__(self, num_steps, num_processes, state_shape, action_shape):
		self.s = torch.zeros(num_steps + 1, num_processes, *state_shape)
		self.r = torch.zeros(num_steps, num_processes, 1)
		self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
		self.returns = torch.zeros(num_steps + 1, num_processes, 1)
		self.log_probs = torch.zeros(num_steps, num_processes, 1)
		self.a = torch.zeros(num_steps, num_processes, 1)  # not + 1 - no action on terminal step

		# need integers, not floats
		self.a = self.a.long()

		# TODO: can we make this congruent with other replay memory???
		self.t = torch.ones(num_steps + 1, num_processes, 1)

		# This is for gym-style envs where a terminal flag is passed for BOTH case when
		# goal is completed and episode length is done
		# We don't want terminal flag when goal is not completed
		self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

		self.num_steps = num_steps
		self.step = 0

	def insert(self, s, a, log_probs, v, r, t, bad_masks):
		self.s[self.step + 1].copy_(s)
		self.a[self.step].copy_(a)
		self.log_probs[self.step].copy_(log_probs)
		self.value_preds[self.step].copy_(v)
		self.r[self.step].copy_(r)
		self.t[self.step + 1].copy_(t)
		self.bad_masks[self.step + 1].copy_(bad_masks)

		self.step = (self.step + 1) % self.num_steps

	def after_update(self):
		self.s[0].copy_(self.s[-1])
		self.t[0].copy_(self.t[-1])
		self.bad_masks[0].copy_(self.bad_masks[-1])

	def compute_returns(self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=True):
		if use_proper_time_limits:
			if use_gae:
				self.value_preds[-1] = next_value
				gae = 0

				for step in reversed(range(self.r.size(0))):
					delta = self.r[step] + gamma * self.value_preds[step + 1] \
							* self.t[step + 1] - self.value_preds[step]

					gae = delta + gamma * gae_lambda * self.t[step + 1] * gae
					gae = gae * self.bad_masks[step + 1]
					self.returns[step]= gae + self.value_preds[step]

			else:
				self.returns[-1] = next_value
				for step in reversed(range(self.r.size(0))):
					self.returns[step] = (self.returns[step + 1] * gamma * self.t[step + 1] + self.r[step]) \
										 * self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.value_preds[step]

		else:
			pass

	def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
		num_steps, num_processes = self.r.size()[:2]
		batch_size = num_processes * num_steps

		if mini_batch_size is None:
			assert batch_size >= num_mini_batch, f'batch_size: {batch_size}, num_mini_batch: {num_mini_batch}'
			mini_batch_size = batch_size // num_mini_batch

		# TODO: this seems like a pretty high-compute-overhead way to sample random numbers....
		sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

		# actions and log_probs do *not* have their last example dropped
		for idxs in sampler:
			s_batch = self.s[:-1].view(-1, *self.s.size()[2:])[idxs]
			a_batch = self.a.view(-1, self.a.size(-1))[idxs]
			value_preds_batch = self.value_preds[:-1].view(-1, 1)[idxs]
			returns_batch = self.returns[:-1].view(-1, 1)[idxs]
			t_batch = self.t[:-1].view(-1, 1)[idxs]
			old_log_probs_batch = self.log_probs.view(-1, 1)[idxs]

			if advantages is None:
				adv_tgt = None
			else:
				adv_tgt = advantages.view(-1, 1)[idxs]

			yield s_batch, a_batch, value_preds_batch, returns_batch, t_batch, old_log_probs_batch, adv_tgt

