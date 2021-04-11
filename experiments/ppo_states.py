import sys
sys.path.append('../')

import time
import argparse
import pickle
from tensorboardX import SummaryWriter
import numpy as np
import torch

from environments.kuka import KukaEnv
from agents.ppo import PPOAgentStates
from utils import FrameStackEnv
from memory import RolloutStorage
from utils import ExperienceGenerator
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--seed', type=int, default=1155, help='random seed')
parser.add_argument('--repeat', type=int, default=25, help='number of times to repeat a given action')
parser.add_argument('--max-ep-len', type=int, default=1000, help='maximum number of steps per episode')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how often to print progress')
parser.add_argument('--episodes', type=int, default=10000, help='number of training episodes to run')
parser.add_argument('--images', action='store_true', default=True, help='Image-based states')
parser.add_argument('--n-steps', type=int, default=500000, help='number of steps for training')
parser.add_argument('--name', help='run name (within the experiment)')
parser.add_argument('--experiment-name', help='experiment name')
args = parser.parse_args()

env = KukaEnv(
    renders=args.render,
    is_discrete=True,
    max_steps=args.max_ep_len,
    action_repeat=args.repeat,
    images=False,
    static_all=True,
    static_obj_rnd_pos=False,
    rnd_obj_rnd_pos=False,
    full_color=False,
	width=84,
	height=84
)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

env = FrameStackEnv(env, 3, 'tensors', 'states')

experience_generator = ExperienceGenerator(env)

agent = PPOAgentStates(
	num_layers=3,
	num_hidden=256,
	state_shape=9 * 3,
	action_shape=7,
	lr=1e-4,
	episodes_per_learning_round=4,
	learning_iterations_per_round=8,
	gamma=0.99,
	clip_eps=0.2,
	clip_grad=5.0,
	experience_generator=experience_generator
)


writer = SummaryWriter(log_dir=f'./{args.experiment_name}/logs/{args.name}')

completed = []
reward_hist = []
state = time.time()
global_steps = 0
episode = 0

start = time.time()

while global_steps < args.n_steps:
	inner_completed, inner_steps = agent.step()

	completed.extend(inner_completed)
	global_steps += inner_steps

	episode += 4

	if episode % 12 == 0:
		print(
			f'Episode: {episode}, Pct: {np.mean(completed[-100:])}, Hours time {(time.time() - start) / 3600}, Step: {global_steps}'
		)
		writer.add_scalar('Completed', np.mean(completed[-100:]), episode)


with open(f'./{args.experiment_name}/{args.name}.data', 'wb') as f:
	pickle.dump(completed, f)




# global traingin steps / number of steps per update / num processes
# num_steps per update also used in <for step in range(5)>
# num_updates = int(args.n_steps) // 5 // 1
#
#
#
#
# while global_steps < args.n_steps:
# 	inner_completed = []
# 	inner_r = []
#
# 	s = env.reset()
# 	rollouts.s[0].copy_(s)
#
# 	done = False
# 	step = 0
#
# 	while not done:
# 		with torch.no_grad():
# 			value, a, log_prob = agent.act(rollouts.s.[step])
#
# for j in range(num_updates):
# 	for step in range(5):
# 		with torch.no_grad():
# 			value, a, log_prob = agent.act(rollouts.s[step])
#
# 		s_, r, picked_up, t, _ = env.step(a)
# 		reward_hist.append(r)
#
# 		# here, bad transition is when we get terminal flag due to max number of steps in episode
# 		# Using this RolloutStorage, we do not do 1 - t, so this is backwards
# 		# TODO: should we make this congruent with the ReplayMemory class?
# 		if t:
# 			if picked_up:
# 				t = torch.FloatTensor([0.0])
# 				bad_mask = torch.FloatTensor([1.0])
# 			else:
# 				t = torch.FloatTensor([1.0])
# 				bad_mask = torch.FloatTensor([0.0])
#
# 		else:
# 			t = torch.FloatTensor([1.0])
# 			bad_mask = torch.FloatTensor([1.0])
#
# 		rollouts.insert(
# 			s, a, log_prob, value, torch.FloatTensor([r]), t, bad_mask
# 		)
#
# 	with torch.no_grad():
# 		next_value = agent.get_value(rollouts.s[-1]).detach()
#
# 	rollouts.compute_returns(
# 		next_value=1,
# 		use_gae=True,
# 		gamma=0.99,
# 		gae_lambda=0.95,
# 		use_proper_time_limits=True
# 	)
#
# 	value_loss, action_loss, dist_entropy = agent.update(rollouts)
#
# 	rollouts.after_update()
#
# 	if j % 10 == 0:
# 		print(np.mean(reward_hist[-500:]))
#

