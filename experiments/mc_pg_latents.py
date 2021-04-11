"""
static_static_gray_overcam -- 249


down_32:
	min: array([-2.84521461, -4.69388723, -2.60996819, -4.82913017, -4.83883858,
       -1.54387987, -8.75399494, -4.57264471,  0.7178188 , -4.11438704,
       -3.56601024, -4.50525665, -4.34889555, -6.07994366, -4.73657751,
       -4.93218231, -5.20519829, -4.13060665, -2.48718953, -4.83062506,
       -5.73168135, -5.31338549, -6.594697  , -6.49887657, -4.22225094,
       -5.28586006, -4.30047178, -3.14535189, -3.37820816, -4.43569708,
       -2.68445635, -4.10923433])
	max:
	array([ 7.7632966 ,  8.19318962,  5.12180281,  0.8992281 ,  4.06726789,
        5.04264784,  0.17909494,  5.968081  ,  6.47219181,  3.44464588,
        4.90059757,  3.80958652,  4.47238636,  6.78564262,  4.93471813,
        6.04560661,  3.40839243,  5.72356415,  7.33014774,  5.58266544,
        1.84771132,  3.84257674,  5.45425797,  2.60517812,  3.36480546,
        4.63301182,  4.54069233,  5.70224237,  2.54797173, -0.72505683,
        6.30836964,  2.98688722])

"""
import sys
sys.path.append('../')

import time
import argparse
import torch
from torch import optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
import numpy as np
from collections import namedtuple
from environments.kuka import KukaEnv
from models import PolicyFC
# from utils import DotDict
from models.encoders import VAE, NatureCNN
from utils import FrameStackEnv
from collections import Counter
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Training simulation for various deep RL environments.')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--seed', type=int, default=19, help='random seed')
parser.add_argument('--repeat', type=int, default=25, help='number of times to repeat a given action')
parser.add_argument('--max-ep-len', type=int, default=1000, help='maximum number of steps per episode')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how often to print progress')
parser.add_argument('--episodes', type=int, default=10000, help='number of training episodes to run')
parser.add_argument('--images', action='store_true', default=False, help='Image-based states')
# parser.add_argument('--name', help='experiment name')
parser.add_argument('--z-dims', type=int, help='the dimensions of z')
parser.add_argument('--encoder-type', type=str, default='none', help='type of encoder to use')
parser.add_argument('--n-steps', type=int, default=100000, help='number of steps for training')
parser.add_argument('--name', help='run name (within the experiment)')
parser.add_argument('--experiment-name', help='experiment name')
args = parser.parse_args()

env = KukaEnv(
	renders=args.render,
	is_discrete=True,
	max_steps=args.max_ep_len,
	action_repeat=args.repeat,
	images=args.images,
	static_all=True,
	static_obj_rnd_pos=True,
	rnd_obj_rnd_pos=False,
	full_color=True
)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

env = FrameStackEnv(env, 3, 'tensors', 'states')

writer = SummaryWriter(log_dir=f'./{args.experiment_name}/logs/{args.name}')

saved_action = namedtuple('saved_action', ['log_prob', 'value'])

model = PolicyFC(args.z_dims)
clip_grad = 20.0

# Encoder
if args.encoder_type == 'vae':
	encoder = VAE(args.z_dims)
	encoder.load_state_dict(torch.load('./models/vae_static_rnd_down_32.pth'))
# elif args.encoder_type == 'stdim':
# 	encoder_features = DotDict({
# 		'feature_size': args.z_dims,
# 		'no_downsample': True,
# 		'method': 'infonce-stdim',
# 		'end_with_relu': False,
# 		'linear': True
# 	})
# 	encoder = NatureCNN(3, encoder_features)
# 	encoder.load_state_dict(torch.load('./models/stdim_static_rnd_32.pth'))
else:
	encoder = None


optimizer = optim.Adam(model.parameters(), lr=3e-4)
eps = np.finfo(np.float32).eps.item()

# FOR NORM'ING THE LATENT CODES
# z_min = np.array([-2.84521461, -4.69388723, -2.60996819, -4.82913017, -4.83883858,
#        -1.54387987, -8.75399494, -4.57264471,  0.7178188 , -4.11438704,
#        -3.56601024, -4.50525665, -4.34889555, -6.07994366, -4.73657751,
#        -4.93218231, -5.20519829, -4.13060665, -2.48718953, -4.83062506,
#        -5.73168135, -5.31338549, -6.594697  , -6.49887657, -4.22225094,
#        -5.28586006, -4.30047178, -3.14535189, -3.37820816, -4.43569708,
#        -2.68445635, -4.10923433])
# z_max = np.array([ 7.7632966 ,  8.19318962,  5.12180281,  0.8992281 ,  4.06726789,
#         5.04264784,  0.17909494,  5.968081  ,  6.47219181,  3.44464588,
#         4.90059757,  3.80958652,  4.47238636,  6.78564262,  4.93471813,
#         6.04560661,  3.40839243,  5.72356415,  7.33014774,  5.58266544,
#         1.84771132,  3.84257674,  5.45425797,  2.60517812,  3.36480546,
#         4.63301182,  4.54069233,  5.70224237,  2.54797173, -0.72505683,
#         6.30836964,  2.98688722])
# z_denom = z_max - z_min


def select_action(state):
	""" Assumes that the state is a numpy array!

	log_prob should be a scalar
	value should be [scalar]

	Args:
		state (ndarray):

	Returns:
		action (int)
	"""

	# the latent vectors we're passing here will be <args.z_dim>
	# as env.images is tied to a TON of other things (what a bad code architecture, huh? My bad...)
	# we can just get rid of the IF/ELSE block

	# if env.images:
	# 	state = transforms.ToTensor()(state).unsqueeze(0).float()
	# 	probs, state_value = model(state)
	# 	probs = probs[0]
	# 	state_value = state_value[0]

	# else:

	# state = torch.from_numpy(state).float()
	probs, state_value = model(state)

	# create a categorical distribution over the list of probabilities of actions
	m = Categorical(probs)

	# sample an action using the above dist
	action = m.sample()

	# save action to buffer
	model.saved_actions.append(saved_action(m.log_prob(action), state_value))

	return action.item()


def finish_episode(report=False):
	"""
	Training code. Calcultes actor and critic loss and performs backprop.
	"""
	R = 0
	saved_actions = model.saved_actions
	policy_losses = []  # list to save actor (policy) loss
	value_losses = []  # list to save critic (value) loss
	returns = []  # list to save the true values

	# calculate the true value using rewards returned from the environment
	for r in model.rewards[::-1]:
		# calculate the discounted value
		R = r + args.gamma * R
		returns.insert(0, R)

	# if np.max(model.rewards) > 10:
	# 	print(f'Before norm: {model.rewards[::-1]}')

	returns = torch.tensor(returns, dtype=torch.float32)
	# returns = (returns - returns.mean()) / (returns.std() + eps)

	# if np.max(model.rewards) > 10:
	# 	print(f'After norm: {returns}')

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

	# clip_grad_norm_(model.parameters(), clip_grad)

	optimizer.step()

	# reset rewards and action buffer
	del model.rewards[:]
	del model.saved_actions[:]

	if report:
		return torch.stack(policy_losses).sum(), torch.stack(value_losses).sum()


def main():
	start = time.time()

	running_reward = 10

	completed = []
	reward_hist = []
	global_steps = 0
	episode = 0
	pi_loss = []
	v_loss = []

	# run inifinitely many episodes
	# for i_episode in count(1):
	# for i in range(args.episodes):
	while global_steps < args.n_steps:

		# reset environment and episode reward
		s = env.reset()

		done = False
		step = 0
		inner_completed = []
		inner_r = []

		while not done:

			# Encoding the state
			if encoder is not None:
				s = encoder.encode_state(
					torch.tensor(np.moveaxis(s, -1, 0), dtype=torch.float).unsqueeze(0)
				).detach().numpy()[0]

			# norming
			# state = (state - z_min) / z_denom

			# ep_reward = 0
			# inner_picked_up = []
			# for each episode, only run some steps so that we don't
			# infinite loop while learning
			a = select_action(s.float())
			s_, r, picked_up, t, _ = env.step(a)

			inner_completed.append(picked_up)
			inner_r.append(r)

			model.rewards.append(r)

			# two exit conditions: (1) if time-limit is reached (2) if object is picked up
			if t:
				done = True
			if picked_up:
				done = True

			s = s_

			step += 1
			global_steps += 1

		pl, vl = finish_episode(True)

		pi_loss.append(pl.item())
		v_loss.append(vl.item())

		if np.sum(inner_completed) > 0:
			completed.append(1)
		else:
			completed.append(0)

		reward_hist.append(np.sum(inner_r))

		episode += 1

		if episode % 10 == 0:
			print(f'Episode {episode}, Pct: {np.mean(completed[-100:])}, R: {np.mean([reward_hist[-100:]])}, Hours time {(time.time() - start) / 3600}')
			writer.add_scalar('Completed', np.mean(completed[-100:]), episode)
			# writer.add_scalar('Alpha', agent.alpha, episode)
			writer.add_scalar('Actor_Losses', np.mean(pi_loss[-100:]), episode)
			writer.add_scalar('Critic_Losses', np.mean(v_loss[-100:]), episode)
			# writer.add_scalar('Qs', np.mean(qs[-100:]), episode)

			# if episode % 1000 == 0:
			# 	print('----------ACTION COUNTER-----------------')
			# 	print(Counter(replay_memory.a.numpy()))
			# 	s, a, r, s_, t = replay_memory.sample(100)
			# 	print(r)
			# 	print(t)
			# 	print('-----------------------------------------')

	return completed


if __name__ == '__main__':
	completed = main()

	with open(f'./{args.experiment_name}/{args.name}.data', 'wb') as f:
		pickle.dump(completed, f)
