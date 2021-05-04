import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1]))
            # kornia.augmentation.RandomCrop((84, 84))
        )

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.eoo = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, eoo):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        np.copyto(self.eoo[self.idx], eoo)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug,
                                         device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)

        obses_aug = self.aug_trans(obses_aug)
        next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug

    def sample_traj(self, batch_size, k):
        """Really will only work for envs with fixed length episodes, such as in dm_control"""
        end_idxs = np.where(self.eoo == 1)[0]

        beg_ranges = end_idxs + 1
        beg_ranges = np.delete(beg_ranges, -1)
        beg_ranges = np.insert(beg_ranges, np.array([0]), 0)

        end_ranges = end_idxs - k

        traj_idxs = []

        n_slots = len(end_idxs)

        for _ in range(batch_size):
            slot_idx = np.random.choice(n_slots)
            beg = np.random.choice(range(beg_ranges[slot_idx], end_ranges[slot_idx]))
            traj_idxs.append([beg + i for i in range(k)])

        actions = np.array([self.actions[traj_idxs[i]] for i in range(batch_size)])
        # obses = np.array([
        #     self.aug_trans(torch.tensor(self.obses[traj_idxs[i]]).float()).numpy() for i in range(batch_size)
        # ])
        # obses_next = np.array([
        #     self.aug_trans(torch.tensor(self.next_obses[traj_idxs[i]]).float()).numpy() for i in range(batch_size)
        # ])
        obses = np.array([
            self.random_crop(self.obses[traj_idxs[i]]) for i in range(batch_size)
        ])
        obses_next = np.array([
            self.random_crop(self.next_obses[traj_idxs[i]]) for i in range(batch_size)
        ])
        rewards = np.array([self.rewards[traj_idxs[i]] for i in range(batch_size)])

        actions = torch.as_tensor(actions, device=self.device)
        obses = torch.tensor(obses, device=self.device).float()
        obses_next = torch.tensor(obses_next, device=self.device).float()
        rewards = torch.tensor(rewards, device=self.device)

        return obses, actions, obses_next, rewards


def random_crop(imgs, out=84):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        img = np.pad(img, 4)
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped
