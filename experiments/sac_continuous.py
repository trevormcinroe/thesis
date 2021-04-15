import sys
sys.path.append('../')

from environments.kuka import KukaEnv
from agents.sac import SACContinuousAgentImages
from memory import ReplayMemory
from utils import FrameStackEnv, FrameStack, eval_mode, sanity_check, eval_agent, eval_representation
import time
import argparse
import torch
import pickle
import numpy as np
from torchvision.transforms import ToTensor
from tqdm import tqdm
from collections import Counter
import gym
from tensorboardX import SummaryWriter
import dmc2gym
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--seed', type=int, default=1155, help='random seed')
parser.add_argument('--repeat', type=int, default=25, help='number of times to repeat a given action')
parser.add_argument('--max-ep-len', type=int, default=1000, help='maximum number of steps per episode')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how often to print progress')
parser.add_argument('--episodes', type=int, default=10000, help='number of training episodes to run')
parser.add_argument('--images', action='store_true', default=True, help='Image-based states')
parser.add_argument('--n-steps', type=int, default=100000, help='number of steps for training')
parser.add_argument('--name', help='experiment name')
parser.add_argument('--experiment-name', help='experiment name')
parser.add_argument('--init-steps', type=int, default=1000)
parser.add_argument('--eval-freq', type=int, default=1000)

args = parser.parse_args()

env = dmc2gym.make(
	domain_name='cartpole',
	task_name='swingup',
	seed=args.seed,
	visualize_reward=False,
	from_pixels=True,
	height=84,
	width=84,
	frame_skip=8
)

env = FrameStack(env, 3)

action_shape = env.action_space.shape[0]
episodes_between_eval = args.eval_freq // env._max_episode_steps

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = SACContinuousAgentImages(
	state_cc=9,
	num_filters=32,
	num_convs=4,
	action_shape=action_shape,
	actor_lr=1e-3,
	actor_beta=0.9,
	critic_lr=1e-3,
	critic_beta=0.9,
	alpha_lr=1e-4,
	alpha_beta=0.5,
	batch_size=128,
	critic_target_update_freq=2,
	actor_update_freq=2,
	critic_update_freq=1,
	critic_tau=0.01,
	init_temperature=0.1,
	gamma=0.99,
	device='cuda:0',
	clip_grad=False,
	encoder_tau=0.05,
	encoder_lr=1e-3
)
# print(sanity_check(agent.actor.encoder.convs, agent.critic.encoder.convs))
replay_memory = ReplayMemory(mem_size=100000, s_shape=[9, 84, 84], a_shape=action_shape, norm=False)

writer = SummaryWriter(log_dir=f'./{args.experiment_name}/logs/{args.name}')

print('collecting...')
n_random = 0
while n_random < args.init_steps:
	s = env.reset()
	s = torch.tensor(s / 255.).float()

	done = False

	steps = 0

	while not done:
		# a = np.random.random(action_shape)
		a = env.action_space.sample()
		s_, r, done, _ = env.step(a)

		steps += 1

		s_ = torch.tensor(s_ / 255.).float()

		# Some infinite bootstrapping
		# i.e., never returns the '1.0' flag for done, since there is not target goal state
		done_bool = 0 if steps == env._max_episode_steps else float(done)

		replay_memory.store(s, torch.tensor(a), r, s_, done_bool)

		s = s_

		n_random += 1

reward_hist = []
eval_hist = []
err_hist = []
start = time.time()
actor_losses = []
critic_losses = []
qs = []
global_steps = 0
episode = 0
eval_episode = 0


if not os.path.isdir(f'./{args.experiment_name}/{args.name}_encoder/'):
	os.mkdir(f'./{args.experiment_name}/{args.name}_encoder/')

print('training...')
while global_steps < args.n_steps:
	inner_completed = []
	inner_r = []

	# It's important that we reset the env before going back into training mode...
	if global_steps % episodes_between_eval == 0:
		print(episode)
		eval_hist.append(eval_agent(env, agent, 10))
		# err_hist.append(eval_representation(agent.critic.encoder, replay_memory))
		writer.add_scalar('Eval reward', np.mean(eval_hist[-5:]), eval_episode)
		# writer.add_scalar('Eval repr', np.mean(err_hist[-5:]), eval_episode)
		eval_episode += 1

		torch.save(
			agent.critic.encoder.state_dict(),
			f'./{args.experiment_name}/{args.name}_encoder/{global_steps}.pt'
		)

	s = env.reset()
	s = torch.tensor(s / 255.).float()
	done = False
	steps = 0

	episode_reward = 0


	while not done:
		with eval_mode(agent):
			a = agent.sample_action(s.unsqueeze(0).float().to('cuda:0'))
		# print(a)
		s_, r, done, _ = env.step(a)
		# print(r)
		steps += 1
		episode_reward += r

		s_ = torch.tensor(s_ / 255.).float()

		agent.update(replay_memory, global_steps, True)

		# Some infinite bootstrapping
		# i.e., never returns the '1.0' flag for done, since there is not target goal state
		done_bool = 0 if steps == env._max_episode_steps else float(done)

		replay_memory.store(s, torch.tensor(a), r, s_, done_bool)

		s = s_

		global_steps += 1

	reward_hist.append(episode_reward)

	episode += 1

	actor_losses.append(np.mean(agent.actor_losses))
	critic_losses.append(np.mean(agent.critic_losses))
	qs.append(np.mean(agent.qs))
	agent.clear_losses()

	# print(sanity_check(agent.actor.encoder.convs, agent.critic.encoder.convs))

	# if episode % 10 == 0:
	print(f'Episode {episode}, R: {np.mean([reward_hist[-5:]])}, Hours time {(time.time() - start) / 3600}, a: {round(agent.alpha.item(), 4)}/{round(agent.log_alpha.item(), 4)}, Step: {global_steps}')
	writer.add_scalar('Reward', np.mean(reward_hist[-5:]), episode)
	writer.add_scalar('Alpha', agent.alpha, episode)
	writer.add_scalar('Actor_Losses', np.mean(actor_losses[-100:]), episode)
	writer.add_scalar('Critic_Losses', np.mean(critic_losses[-100:]), episode)
	writer.add_scalar('Qs', np.mean(qs[-100:]), episode)


	# if episode % 10 == 0:
	# 	s, a, r, s_, t = replay_memory.sample(100)
	# 	print(a)
	# 	print(t)


with open(f'./{args.experiment_name}/{args.name}_training.data', 'wb') as f:
	pickle.dump(reward_hist, f)

with open(f'./{args.experiment_name}/{args.name}_eval.data', 'wb') as f:
	pickle.dump(eval_hist, f)

with open(f'./{args.experiment_name}/{args.name}_repr.data', 'wb') as f:
	pickle.dump(err_hist, f)
