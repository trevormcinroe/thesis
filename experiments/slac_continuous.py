import argparse
import os
from datetime import datetime

import torch

from agents.slac import SlacAlgorithm, make_dmc, Trainer

def main(args):
    env = make_dmc(
        domain_name=args.domain_name,
        task_name=args.task_name,
        action_repeat=args.action_repeat,
        image_size=64,
    )
    env_test = make_dmc(
        domain_name=args.domain_name,
        task_name=args.task_name,
        action_repeat=args.action_repeat,
        image_size=64,
    )

    log_dir = '/media/trevor/mariadb/thesis'

    algo = SlacAlgorithm(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_repeat=args.action_repeat,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
    )
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        seed=args.seed,
        num_steps=args.num_steps,
        name=args.name
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=2 * 10 ** 6)
    parser.add_argument("--domain_name", type=str, default="cheetah")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument('--name', type=str)
    parser.add_argument('--experiment-name', type=str, default='bounds')
    args = parser.parse_args()

    data_root = '/media/trevor/mariadb/thesis'

    if not os.path.isdir(f'{data_root}/{args.experiment_name}'):
        os.mkdir(f'{data_root}/{args.experiment_name}')

    if not os.path.isdir(f'{data_root}/{args.experiment_name}/{args.name}/'):
        os.mkdir(f'{data_root}/{args.experiment_name}/{args.name}/')

    main(args)