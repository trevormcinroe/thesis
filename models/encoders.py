import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms

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


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class SACAEEncoderDiscrete(nn.Module):
    def __init__(self, state_cc, feature_dim, num_filters, device='cpu'):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(state_cc, num_filters, (7, 7), 3),
            nn.Conv2d(num_filters, num_filters, (5, 5), 2),
            nn.Conv2d(num_filters, num_filters, (5, 5), 2),
        ])

        self.d = nn.Linear(512, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

        self.to(device)

    def forward(self, s, detach=False):
        for conv in self.convs:
            s = F.relu(conv(s))

        s = s.view(s.size(0), -1)

        if detach:
            s = s.detach()

        s = self.ln(self.d(s))

        return torch.tanh(s)

    def copy_conv_weights_from(self, source):
        """Authors use this to tie weights between Actor and Critic encoders. (?)"""
        for i in range(3):
            tie_weights(src=source.convs[i], tgt=self.convs[i])





class SACAEEncoder(nn.Module):
    def __init__(self, state_cc, feature_dim, num_filters, device='cpu'):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(state_cc, num_filters, (7, 7), 2),
                nn.Conv2d(num_filters, num_filters, (5, 5), 3),
                nn.Conv2d(num_filters, num_filters, (5, 5), 3)
            ]
        )

        self.d1 = nn.Linear(512, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

        self.to(device)

    def reparameterize(self):
        """This seems to not be used. In the paper, the authors mention a deterministic VAE. This is probably why"""
        pass

    def forward_conv(self):
        """Authors define this operation but it isn't used anywhere, really..."""
        pass


    def forward(self, x, detach=False):

        for conv in self.convs:
            x = F.relu(conv(x))

        x = x.view(x.size(0), -1)

        if detach:
            x = x.detach()

        x = self.ln(self.d1(x))

        return torch.tanh(x)

    def copy_conv_weights_from(self, source):
        """Authors use this to tie weights between Actor and Critic encoders. (?)"""
        for i in range(3):
            tie_weights(src=source.convs[i], tgt=self.convs[i])


    def log(self):
        """Should each model have their own logger? Probably not..."""
        pass


# class VAE(nn.Module):
#     def __init__(self, z_dims, state_cc, num_filters, device='cpu'):
#         super().__init__()
#         self.z_dims = z_dims
#
#
#
#         ## ENCODER ##
#         self.encoder = nn.Sequential(
#             nn.Conv2d(state_cc, num_filters, (7, 7), 3, bias=False),
#             nn.BatchNorm2d(num_filters),
#             nn.ReLU(),
#             nn.Conv2d(num_filters, num_filters, (5, 5), 2, bias=False),
#             nn.BatchNorm2d(num_filters),
#             nn.ReLU(),
#             nn.Conv2d(num_filters, num_filters, (5, 5), 2, bias=False),
#             nn.BatchNorm2d(num_filters),
#             nn.ReLU(),
#         )
#         self.mu_fc = nn.Linear(512, self.z_dims)
#         self.var_fc = nn.Linear(512, self.z_dims)
#
#         ## DECODER ##
#         self.decoder_input = nn.Linear(self.z_dims, 512)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(32, num_filters, (5, 5), 2, bias=False),
#             nn.BatchNorm2d(num_filters),
#             nn.ReLU(),
#             nn.ConvTranspose2d(num_filters, num_filters, (5, 5), 2, bias=False),
#             nn.BatchNorm2d(num_filters),
#             nn.ReLU(),
#             nn.ConvTranspose2d(num_filters, num_filters, (5, 5), 3, bias=False),
#             nn.BatchNorm2d(num_filters),
#             nn.ReLU(),
#             nn.ConvTranspose2d(num_filters, state_cc, (8, 8), 1, bias=False),
#             nn.Sigmoid()
#         )
#
#         self.to(device)
#
#     def encode(self, x):
#         x = self.encoder(x)
#         x = torch.flatten(x, start_dim=1)
#         mu = self.mu_fc(x)
#         var = self.var_fc(x)
#         return mu, var
#
#     def reparam(self, mu, var):
#         std = torch.exp(var / 2)
#         e = torch.randn_like(std)
#         return e * std + mu
#
#     def decode(self, z):
#         x = self.decoder_input(z)
#         x = x.view(-1, 32, 4, 4)
#         x = self.decoder(x)
#         return x
#
#     def encode_state(self, x):
#         mu, var = self.encode(x)
#         z = self.reparam(mu, var)
#         return z
#
#     def forward(self, x):
#         mu, var = self.encode(x)
#         z = self.reparam(mu, var)
#         x_hat = self.decode(z)
#         return x_hat, mu, var
#

class VAE(nn.Module):
    def __init__(self, z_dims, state_cc, num_filters, device='cpu'):
        super().__init__()
        self.z_dims = z_dims



        ## ENCODER ##
        self.encoder = nn.Sequential(
            nn.Conv2d(state_cc, num_filters, (7, 7), 3, bias=False),
            # nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, (5, 5), 2, bias=False),
            # nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, (5, 5), 2, bias=False),
            # nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.ln = nn.LayerNorm(512)
        self.mu_fc = nn.Linear(512, self.z_dims)
        self.var_fc = nn.Linear(512, self.z_dims)

        ## DECODER ##
        self.decoder_input = nn.Linear(self.z_dims, 512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, num_filters, (5, 5), 2, bias=False),
            # nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, num_filters, (5, 5), 2, bias=False),
            # nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, num_filters, (5, 5), 3, bias=False),
            # nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, state_cc, (8, 8), 1, bias=False),
            nn.Sigmoid()
        )

        self.to(device)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.ln(x)
        mu = self.mu_fc(x)
        var = self.var_fc(x)
        return mu, var

    def reparam(self, mu, var):
        std = torch.exp(var / 2)
        e = torch.randn_like(std)
        return torch.tanh(e * std + mu)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 32, 4, 4)
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


def vae_loss(x, x_hat, mu, var, weight):
    # Reconstruction error
    # recon_err = F.binary_cross_entropy(x_hat, x, reduction='sum')
    recon_err = F.smooth_l1_loss(x_hat, x, reduction='sum')

    # KL
    kl_div = torch.mean(
        -0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1), dim=0
    )

    return recon_err + weight * kl_div


def train_vae(model, optimizer, replay_memory, n_epochs, batch_size, seed, show=False):
    for i in tqdm(range(n_epochs)):
        s, _, _, _, _ = replay_memory.sample(batch_size)

        x = []

        for img in s:
            noisy = img.numpy() + np.random.normal(loc=0.0, scale=0.1, size=img.shape)
            x.append(np.clip(noisy, 0.0, 1.0))

        x = np.array(x)
        x = torch.tensor(x).float()

        preds, mu, var = model(x.to('cuda:0'))
        loss = vae_loss(s.to('cuda:0'), preds, mu, var, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i % 100 == 0 & show:
            img = torch.cat([
                x[0][0, :, :].unsqueeze(0),
                preds[0][0, :, :].cpu().unsqueeze(0),
                s[0][0, :, :].unsqueeze(0)
            ], dim=2)

            transforms.ToPILImage()(img).save(f'./imgs/vae_{seed}_training_{i}.png')


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, output_logits=False):
        super().__init__()

        # assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape, num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)

# a = PixelEncoder(obs_shape=[9, 84, 84], feature_dim=50, num_layers=4, num_filters=32, output_logits=True)
# print(a)