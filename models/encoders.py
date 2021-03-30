import torch
from torch import nn
import torch.nn.functional as F
# from a2c_ppo_acktr.utils import init


class VAE(nn.Module):
    def __init__(self, z_dims):
        super().__init__()
        self.z_dims = z_dims

        ## ENCODER ##
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=3, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=3, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.mu_fc = nn.Linear(6272, self.z_dims)
        self.var_fc = nn.Linear(6272, self.z_dims)

        ## DECODER ##
        self.decoder_input = nn.Linear(self.z_dims, 6272)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=3, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=3, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.mu_fc(x)
        var = self.var_fc(x)
        return mu, var

    def reparam(self, mu, var):
        std = torch.exp(var / 2)
        e = torch.randn_like(std)
        return e * std + mu

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 128, 7, 7)
        x = self.decoder(x)
        return x

    def encode_state(self, x):
        mu, var = self.encode(x)
        z = self.reparam(mu, var)
        return z

    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparam(mu, var)
        x_hat = self.decode(z)
        return x_hat, mu, var

# test = VAE(64)
#
# n = torch.rand(size=(1, 1, 224, 224), dtype=torch.float)
#
# xhat, _, _ = test(n)
# print(xhat.shape)


class Flatten(nn.Module):
    def forward(self, x):
        # print(x.view(x.size(0), -1).shape)
        return x.view(x.size(0), -1)


def _init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class NatureCNN(nn.Module):

    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        self.hidden_size = self.feature_size
        self.downsample = not args.no_downsample
        self.input_channels = input_channels
        self.end_with_relu = args.end_with_relu
        self.args = args
        init_ = lambda m: _init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.flatten = Flatten()

        if self.downsample:
            self.final_conv_size = 32 * 7 * 7
            self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        else:
            self.final_conv_size = 6400
            self.final_conv_shape = (64, 9, 6)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 128, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(128, 64, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                #nn.ReLU()
            )
        self.train()

    @property
    def local_layer_depth(self):
        return self.main[4].out_channels

    def forward(self, inputs, fmaps=False):
        f5 = self.main[:6](inputs)
        f7 = self.main[6:8](f5)
        out = self.main[8:](f7)
        if self.end_with_relu:
            assert self.args.method != "vae", "can't end with relu and use vae!"
            out = F.relu(out)
        if fmaps:
            return {
                'f5': f5.permute(0, 2, 3, 1),
                'f7': f7.permute(0, 2, 3, 1),
                'out': out
            }
        return out

    def encode_state(self, inputs):
        return self.forward(inputs=inputs)

# from utils import DotDict
# encoder_features = DotDict({
#         'feature_size': 32,
#         'no_downsample': True,
#         'method': 'infonce-stdim',
#         'end_with_relu': False,
#         'linear': True
# })
# a = NatureCNN(3, encoder_features)
#
# a(torch.rand((32, 3, 224, 224)), fmaps=True)