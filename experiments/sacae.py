import sys
sys.path.append('../')

from environments.kuka import KukaEnv
from agents.sac import SACAEAgentDiscrete
from memory import ReplayMemory
from utils import FrameStackEnv
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

# TODO: modify the DONE flag for off-policy algorithms. If the DONE flag comes in due to a time-limit, there are
# known convergence issues with off-policy algos!
# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/sac.py#L301
# ^ OpenAI does this too

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

env = KukaEnv(
    renders=args.render,
    is_discrete=True,
    max_steps=args.max_ep_len,
    action_repeat=args.repeat,
    images=True,
    static_all=False,
    static_obj_rnd_pos=True,
    rnd_obj_rnd_pos=False,
    full_color=False,
	width=84,
	height=84
)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

env = FrameStackEnv(env, 3, 'tensors')

agent = SACAEAgentDiscrete(
    state_cc=3,
    action_shape=7,
    num_layers=2,
    num_hidden=256,
    gamma=0.99,
    alpha_lr=3e-6,
    alpha_beta=0.9,
    actor_lr=1e-4,
    actor_beta=0.9,
    critic_update_freq=1,
    actor_update_freq=8,
    critic_lr=3e-4,
    critic_beta=0.9,
    critic_tau=0.005,
    critic_target_update_freq=4,
    z_dim=32,
    encoder_lr=1e-3,
    encoder_tau=0.05,
    decoder_lr=1e-3,
    decoder_update_freq=1,
    decoder_latent_lambda=1e-6,
    decoder_weight_lambda=1e-7,
    num_filters=32,
    device='cuda:0',
    clip_grad=5.0,
    batch_size=256
)

replay_memory = ReplayMemory(mem_size=200000, s_shape=[3, 84, 84], norm=False)

writer = SummaryWriter(log_dir=f'./{args.experiment_name}/logs/{args.name}')

# exploration
N_EXPLORE = 100
for _ in tqdm(range(N_EXPLORE)):
	s = env.reset()

	done = False

	while not done:
		a = np.random.choice(7, 1)[0]
		s_, r, picked_up, t, _ = env.step(a)
		# replay_memory.store(s, a, r, s_, t)

		if t:
			if picked_up:
				# replay_memory.store(s, a, r, s_, 1)
				t = 1
				done = True
			else:
				# replay_memory.store(s, a, r, s_, 0)
				t = 0
				done = True

		if picked_up:
			t = 1
			done = True

		replay_memory.store(s, a, r, s_, t)

		s = s_

completed = []
reward_hist = []
start = time.time()
actor_losses = []
critic_losses = []
qs = []
global_steps = 0
episode = 0

# In the original paper, they do 1000 steps of decoder updating during the
# first step in actual learning (beyond exploration)
print('Pre-training the decoder...')
for i in tqdm(range(1000)):
	agent.update(replay_memory, i)

while global_steps < args.n_steps:
	inner_completed = []
	inner_r = []
	s = env.reset()
	done = False
	step = 0

	while not done:
		a = agent.sample_action(s.unsqueeze(0).float().to('cuda:0'))
		s_, r, picked_up, t, _ = env.step(a)
		inner_completed.append(picked_up)
		inner_r.append(r)

		agent.update(replay_memory, step, True)

		if t:
			if picked_up:
				t = 1
				done = True
			else:
				t = 0
				done = True

		if picked_up:
			t = 1
			done = True

		replay_memory.store(s, a, r, s_, t)

		s = s_

		step += 1
		global_steps += 1

	if np.sum(inner_completed) > 0:
		completed.append(1)
	else:
		completed.append(0)

	reward_hist.append(np.sum(inner_r))

	episode += 1

	actor_losses.append(np.mean(agent.actor_losses))
	critic_losses.append(np.mean(agent.critic_losses))
	qs.append(np.mean(agent.qs))
	agent.clear_losses()

	if episode % 10 == 0:
		print(f'Episode {episode}, Pct: {np.mean(completed[-100:])}, R: {np.mean([reward_hist[-100:]])}, Hours time {(time.time() - start) / 3600}, a: {round(agent.alpha.item(), 4)}/{round(agent.log_alpha.item(), 4)}, Step: {global_steps}')
		writer.add_scalar('Completed', np.mean(completed[-100:]), episode)
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