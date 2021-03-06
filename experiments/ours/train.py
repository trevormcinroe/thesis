import sys
sys.path.append('../')
sys.path.append('../../')

import copy
import math
import os
import pickle as pkl
import sys
import time

import numpy as np
import pickle
import dmc2gym
from experiments import dmc2gym_noisy
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder
import data_augs as rad
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env == 'point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    elif cfg.env == 'cartpole_two_poles':
        domain_name = 'cartpole'
        task_name = 'two_poles'
    elif cfg.env == 'cartpole_three_poles':
        domain_name = 'cartpole'
        task_name = 'three_poles'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'quadruped' else 0

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=False,
                       from_pixels=True,
                       height=cfg.image_size,
                       width=cfg.image_size,
                       frame_skip=cfg.action_repeat,
                       camera_id=camera_id)

    # env = dmc2gym_noisy.make(
    #     domain_name=domain_name,
    #     task_name=task_name,
    #     resource_files='../../../../../experiments/distractors/images/*.mp4',
    #     img_source='video',
    #     total_frames=10000,
    #     seed=cfg.seed,
    #     visualize_reward=False,
    #     from_pixels=True,
    #     height=84,
    #     width=84,
    #     frame_skip=cfg.action_repeat,
    #     camera_id=camera_id
    # )

    env = utils.FrameStack(env, k=cfg.frame_stack)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = '/media/trevor/mariadb/thesis/'
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad,
                                          self.device,
                                          self.cfg.env)

        # obs_shape = (3 * 3, 84, 84)
        # pre_aug_obs_shape = (3 * 3, 100, 100)
        #
        # self.replay_buffer = ReplayBuffer(
        #     obs_shape=pre_aug_obs_shape,
        #     action_shape=self.env.action_space.shape,
        #     capacity=cfg.replay_buffer_capacity,
        #     batch_size=cfg.batch_size,
        #     device=self.device,
        #     image_size=84,
        #     pre_image_size=100,
        # )

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        eps_reward = []

        eps_done = 0

        # while eps_done < self.cfg.num_eval_episodes:
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            # self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)

                # This is unnecessary here...
                self.agent.osl.train(True)

                obs, reward, done, info = self.env.step(action)
                # self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1

            # if episode_reward > 0:
            #     eps_reward.append(episode_reward)
            #     average_episode_reward += episode_reward
            #     eps_done += 1
            # else:
            #     continue

            average_episode_reward += episode_reward
            # self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        sd_episode_reward = np.std(eps_reward)
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)
        return average_episode_reward, sd_episode_reward

    def run(self):
        print(f'Eval freq: {self.cfg.eval_frequency}')
        print(f'k: {self.agent.k}')
        print(f'lr: {self.cfg.lr}')

        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()

        if self.cfg.p:

            print('collecting...')
            for _ in tqdm(range(10000)):
                if done:
                    obs = self.env.reset()
                    done = False
                    episode_step = 0

                action = self.env.action_space.sample()
                next_obs, reward, done, info = self.env.step(action)

                done = float(done)
                done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done

                if done:
                    eeo = 1
                else:
                    eeo = 0

                episode_reward += reward

                self.replay_buffer.add(obs, action, reward, next_obs, done,
                                       done_no_max, eeo)
                obs = next_obs
                episode_step += 1

            print('pre-training...')
            for i in tqdm(range(25000)):
                self.agent.pretrain(self.replay_buffer, i)

            # reset replay buffer?
            self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                              self.env.action_space.shape,
                                              100000,
                                              self.cfg.image_pad,
                                              self.device,
                                              self.cfg.env)

        eval_mean = []
        eval_sd = []

        while self.step < (self.cfg.num_train_steps // self.cfg.action_repeat):
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)

                    means, sds = self.evaluate()
                    eval_mean.append(means)
                    eval_sd.append(sds)

                    print(f'OSL: {np.mean(self.agent.osl_loss_hist[-20000:])}')
                    # torch.save(
                    #     self.agent.critic.encoder.state_dict(),
                    #     f'/media/trevor/mariadb/thesis/msl_cartpole_encoder_{self.step * self.cfg.action_repeat}.pt'
                    # )

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                # TODO: at the very top, episode_step is init to 1 but here it is 0...
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            self.agent.osl.train(True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            # TODO: shouldn't DONE always be 0? replay buffer is NOT DONE when adding...
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            if done:
                eeo = 1
            else:
                eeo = 0

            # done_no_max should always be 0, right?
            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max, eeo)

            obs = next_obs
            episode_step += 1
            self.step += 1

        with open(f'/media/trevor/mariadb/thesis/ksl-r-{self.cfg.env}-s{self.cfg.seed}-b{self.cfg.batch_size}-k{self.cfg.agent.params.k}-p{self.cfg.p}-mean.data', 'wb') as f:
            pickle.dump(eval_mean, f)

        # with open(f'/media/trevor/mariadb/thesis/msl-drq-{self.cfg.env}-s{self.cfg.seed}-b{self.cfg.batch_size}-k{self.cfg.agent.params.k}-p{self.cfg.p}-sd.data', 'wb') as f:
        #     pickle.dump(eval_sd, f)
        #
        # self.agent.save(
        #     dir='/media/trevor/mariadb/thesis/',
        #     extras=f'msl-drq-{self.cfg.env}-s{self.cfg.seed}-b{self.cfg.batch_size}-k{self.cfg.agent.params.k}-p{self.cfg.p}-noks-500k'
        # )


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    from train import Workspace as W
    workspace = W(cfg)


    # time.sleep(60*60*4)
    # import time
    # print('waiting...')
    # time.sleep(4 * 60 * 60)
    # print('done waiting!')

    workspace.run()


if __name__ == '__main__':
    main()
