from environments.kuka import KukaEnv
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import time
from PIL import Image
from models.encoders import VAE
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


env = KukaEnv(images=True, static_all=False, is_discrete=True,
			  static_obj_rnd_pos=False, rnd_obj_rnd_pos=False, renders=False,
			  full_color=True)

vae = VAE(32)

vae.to('cuda:0')
optimizer = torch.optim.Adam(vae.parameters(), lr=0.0002)

def vae_loss(x, x_hat, mu, var, weight):
	# Reconstruction error
	recon_err = F.binary_cross_entropy(x_hat, x, reduction='sum')

	# KL
	kl_div = torch.mean(
		-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1), dim=0
	)

	return recon_err + weight * kl_div

memories = np.zeros((10000, 224, 224, 3))
actions = [1, 2, 3, 4, 5, 5, 5, 5, 5]
mem_cntr = 0

while mem_cntr < memories.shape[0]:
	s = env.reset()

	for i in range(600):
		a = np.random.choice(actions)
		state, reward, picked_up, done, _ = env.step(a)
		memories[mem_cntr] = state
		mem_cntr += 1

		if mem_cntr >= memories.shape[0]:
			break

		if done:
			break

EPOCHS = 20000
loss_hist = []
batch_size = 32
choices = [x for x in range(memories.shape[0])]
for i in tqdm(range(EPOCHS)):
	idxes = np.random.choice(choices, batch_size, replace=False)
	x = memories[idxes]
	y = []
	for img in x:
		noisy = img + np.random.normal(loc=0.0, scale=0.1, size=img.shape)
		y.append(np.clip(noisy, 0, 1))

	y = np.array(y)

	x = torch.tensor(np.moveaxis(x, -1, 1), dtype=torch.float)
	y = torch.tensor(np.moveaxis(y, -1, 1), dtype=torch.float)
	# print(y.shape)

	optimizer.zero_grad()

	preds, mu, var = vae(y.to('cuda:0'))
	loss = vae_loss(x.to('cuda:0'), preds, mu, var, 1)
	loss_hist.append(loss.item())
	loss.backward()

	optimizer.step()

	# if i % 1000 == 0:
	# 	print(f'Epoch: {i}, Loss: {np.mean(loss_hist[len(loss_hist)-100:])}')


	if i % 1000 == 0:
		img = torch.cat([
			x[0],
			preds[0].cpu(),
			y[0]
		], dim=2)

		transforms.ToPILImage()(img).save(f'./imgs/out_rnd_static_down_32_{i}.png')


torch.save(vae.state_dict(), './models/vae_rnd_static_down_32.pth')



# print(memories[999])

# for i in range(mem_cntr):
# 	img = memories[i]
# 	img += np.random.normal(loc=0.0, scale=0.1, size=img.shape)
# 	img = np.clip(img, 0, 1)
# 	Image.fromarray((img * 255).astype(np.uint8)).convert('RGB').save(f'../imgs/{i}.png')
# plt.imshow(memories[999])



import time
# print(env.urdf_root)
# actions = [x for x in range(7)]
#
# # view_mat = p.computeViewMatrix(
# # 	cameraEyePosition=[0, 0, 10],
# # 	cameraTargetPosition=[0, 0, 0],
# # 	cameraUpVector=[0, 0.75, 0.75]
# # )
#
# # print(view_mat)
# s = env.reset()
# s = s.dot([0.07, 0.72, 0.21])
# # print(s)
# # Image.fromarray((np.array(s) * 255.).astype(np.uint8)).convert('RGB').save('../imgs/s_color.png')
# Image.fromarray(np.array(s) * 255.).convert("L").save('../imgs/s2.png')
# # im.save('../imgs/s.png')
#
# # print(s.shape)
# for _ in range(1000):
# 	env.reset()
# 	print(env.rnd_obj)
# 	for k in range(50):
# 		env.step(0)


# for _ in range(100):
# 	print('---------------')
# 	s = env.reset()
# 	for j in range(1000):
# 		_, r, _, _, _ = env.step(np.random.choice(actions))
# 		print(f'{j}: {r}')

	# time.sleep(3)

	# plt.imshow(s)
# 	plt.savefig('./out.png')