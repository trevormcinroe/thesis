import torch
from torch import nn
import torch.nn.functional as F


class SACAEDecoder(nn.Module):
	"""feature_dim = z-dim"""
	def __init__(self, state_cc, feature_dim, num_filters, device='cpu'):
		super().__init__()

		self.num_filters = num_filters

		self.fc = nn.Linear(feature_dim, 512)

		self.deconvs = nn.ModuleList(
			[
				nn.ConvTranspose2d(2, num_filters, (5, 5), 3),
				nn.ConvTranspose2d(num_filters, num_filters, (7, 7), 2),
				nn.ConvTranspose2d(num_filters, num_filters, (7, 7), 2),
				nn.ConvTranspose2d(num_filters, num_filters, (7, 7), 1),
				nn.ConvTranspose2d(num_filters, state_cc, (4, 4), 1)
			]
		)

		self.to(device)

	def forward(self, x):
		x = F.relu(self.fc(x))
		x = x.view(-1, 2, 16, 16)
		for deconv in self.deconvs:
			x = F.relu(deconv(x))

		return torch.sigmoid(x)


# SACAEDecoder(12, 32, 32)(torch.rand(1, 1, 32))