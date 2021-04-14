import sys
sys.path.append('../')

from environments.kuka import KukaEnv
from agents.sac import SACContinuousAgentImages
from memory import ReplayMemory
from utils import FrameStackEnv, FrameStack
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
args = parser.parse_args()

env = dmc2gym.make(
        domain_name='cartpole',
        task_name='swingup',
        seed=123,
        visualize_reward=False,
        from_pixels=True,
        height=84,
        width=84,
        frame_skip=1
)

env = FrameStack(env, 3)

action_shape = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = SACContinuousAgentImages(
	state_cc=9,
	num_filters=32,
	num_convs=3,
	action_shape=action_shape,
	actor_lr=1e-3,
	actor_beta=0.9,
	critic_lr=1e-3,
	critic_beta=0.9,
	alpha_lr=1e-3,
	alpha_beta=0.9,
	batch_size=256,
	critic_target_update_freq=2,
	actor_update_freq=2,
	critic_update_freq=1,
	critic_tau=0.005,
	init_temperature=0.01,
	gamma=0.99,
	device='cuda:0',
	clip_grad=False
)

replay_memory = ReplayMemory(mem_size=100000, s_shape=[9, 84, 84], a_shape=action_shape, norm=False)

writer = SummaryWriter(log_dir=f'./{args.experiment_name}/logs/{args.name}')

N_EXPLORE = 100
for _ in tqdm(range(N_EXPLORE)):
	s = env.reset()
	s = torch.tensor(s / 255.).float()

	done = False

	steps = 0

	while not done:
		a = np.random.random(action_shape)
		s_, r, t, _ = env.step(a)
		s_ = torch.tensor(s_ / 255.).float()

		if t:
			done = True

		replay_memory.store(s, torch.tensor(a), r, s_, t)

		s = s_

		steps += 1
		# print(steps)

reward_hist = []
start = time.time()
actor_losses = []
critic_losses = []
qs = []
global_steps = 0
episode = 0

while global_steps < args.n_steps:
	inner_completed = []
	inner_r = []
	s = env.reset()
	s = torch.tensor(s / 255.).float()
	done = False
	step = 0

	while not done:
		a = agent.sample_action(s.unsqueeze(0).float().to('cuda:0'))
		s_, r, t, _ = env.step(a)
		s_ = torch.tensor(s_ / 255.).float()
		inner_r.append(r)

		agent.update(replay_memory, step, True)

		if t:
			done = True

		replay_memory.store(s, torch.tensor(a), r, s_, t)

		s = s_

		step += 1
		# print(step)
		global_steps += 1

	reward_hist.append(np.sum(inner_r))

	episode += 1

	actor_losses.append(np.mean(agent.actor_losses))
	critic_losses.append(np.mean(agent.critic_losses))
	qs.append(np.mean(agent.qs))
	agent.clear_losses()

	# if episode % 10 == 0:
	print(f'Episode {episode}, R: {np.mean([reward_hist[-100:]])}, Hours time {(time.time() - start) / 3600}, a: {round(agent.alpha.item(), 4)}/{round(agent.log_alpha.item(), 4)}, Step: {global_steps}')
	writer.add_scalar('Reward', np.mean(reward_hist[-100:]), episode)
	writer.add_scalar('Alpha', agent.alpha, episode)
	writer.add_scalar('Actor_Losses', np.mean(actor_losses[-100:]), episode)
	writer.add_scalar('Critic_Losses', np.mean(critic_losses[-100:]), episode)
	writer.add_scalar('Qs', np.mean(qs[-100:]), episode)

	if episode % 1000 == 0:
		print('----------ACTION COUNTER-----------------')
		print(Counter(replay_memory.a.numpy()))
		s, a, r, s_, t = replay_memory.sample(100)
		print(r)
		print(t)
		print('-----------------------------------------')


with open(f'./{args.experiment_name}/{args.name}.data', 'wb') as f:
	pickle.dump(reward_hist, f)