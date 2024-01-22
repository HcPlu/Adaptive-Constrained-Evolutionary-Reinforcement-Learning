# -*- coding:utf-8 _*-
"""
=================================================
@Project -> File ：evolution_strategies_openai -> main.py
@Author ：HcPlu
@Version: 1.0
@Date: 2022/11/30 14:34
@@Description: 
==================================================
"""
import argparse
from es.training import run_experiment

# example_config = {
#     # "experiment_name": "agv-breakdown-v1-test",
#     "plot_path": "plots/",
#     "model_path": "models/", # optional
#     "log_path": "logs/", # optional
#
#     # "env": "agv-breakdown-v1",
#     # "n_sessions": 128,
#     # "env_steps": 1600,
#     "population_size": 256,
#     "learning_rate": 0.06,
#     "noise_std": 0.1,
#     "noise_decay": 0.99, # optional
#     "lr_decay": 1.0, # optional
#     "decay_step": 20, # optional
#     "eval_step": 1,
#     "hidden_sizes": (128, 128)
#   }



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default="agv-breakdown-v1-test")
    parser.add_argument('--env', type=str, default="agv-breakdown-v1")
    parser.add_argument('--seed', type=int, default=4213)
    parser.add_argument('--n_sessions', type=int, default=128)
    parser.add_argument('--env_steps', type=int, default=1600)
    parser.add_argument('--population_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.06)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--noise_decay', type=float, default=0.99)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--decay_step', type=int, default=50)
    parser.add_argument('--hidden_sizes', type=int, nargs='*', default=[128,128])

    parser.add_argument('--test-num', type=int, default=80)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.0)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--verbose', type=int, default=True)

    parser.add_argument('--mode',type=str,default="random",choices = ['single','random', 'per', 'ucb'])

    parser.add_argument('--sr', type=int, default=True)
    parser.add_argument('--rank', type=int, default=True)
    # parser.add_argument('--ucb', type=int, default=True)
    parser.add_argument('--cost_limit',type=float,default=50)


    parser.add_argument("--train_instance", type=int, default=0)

    args = parser.parse_known_args()[0]
    return args


policy = run_experiment(get_args())



# to render policy perfomance
