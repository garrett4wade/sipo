from typing import Dict
import argparse
import os
import yaml


def get_base_config():
    parser = argparse.ArgumentParser(
        description='faster_mappo',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--config", type=str)
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument(
        "--user_name",
        type=str,
        default='garrett4wade',
        help=
        "[for wandb usage], to specify user's name for simply collecting training data."
    )
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="")

    parser.add_argument("--sample_reuse", type=int, default=1)

    parser.add_argument("--n_iterations", type=int, default=1)
    parser.add_argument("--archive_policy_dirs", type=str, default=[], nargs="+")
    parser.add_argument("--archive_traj_dirs", type=str, default=[], nargs="+")
    parser.add_argument("--ll_max", type=float, default=100.0)
    parser.add_argument("--threshold_eps", type=float, default=1.0)
    parser.add_argument("--intrinsic_reward_scaling", type=float, default=1.0)

    parser.add_argument("--rbf_gamma", type=float, default=1.0)
    parser.add_argument("--rkhs_action", action='store_true')
    parser.add_argument("--warmup_fraction", type=float, default=0.0)
    parser.add_argument("--inherit_policy", action='store_true')
    parser.add_argument("--eval_episode_length", type=int, default=100)

    parser.add_argument("--warm_up_rate", type=float, default=0.0)

    ################ rspo hyperparameters begin ################
    parser.add_argument("--threshold_annealing_schedule",
                        type=str,
                        default=None)
    parser.add_argument("--auto_alpha", type=float, default=None)
    parser.add_argument("--likelihood_alpha", type=float)
    parser.add_argument("--likelihood_threshold", type=float)
    parser.add_argument("--prediction_reward_alpha", type=float)
    parser.add_argument("--exploration_reward_alpha", type=float)
    parser.add_argument("--use_reward_predictor", action='store_true')
    parser.add_argument("--reward_prediction_multiplier", type=float)
    parser.add_argument("--no_exploration_rewards", action='store_true')
    parser.add_argument("--exploration_threshold", type=float, default=0.6)
    ################# rspo hyperparameters end #################

    # prepare parameters
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="Random seed for numpy/torch")
    parser.add_argument("--seed_specify",
                        action="store_true",
                        default=False,
                        help="Random or specify seed for numpy/torch")
    parser.add_argument("--runing_id",
                        type=int,
                        default=1,
                        help="the runing index of experiment")
    parser.add_argument(
        "--cuda",
        action='store_false',
        default=True,
        help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument(
        "--cuda_deterministic",
        action='store_false',
        default=True,
        help=
        "by default, make sure random seed effective. if set, bypass such function."
    )
    parser.add_argument("--n_training_threads",
                        type=int,
                        default=1,
                        help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=None)
    parser.add_argument("--n_eval_rollout_threads", type=int, default=None)
    parser.add_argument("--num_trian_envs", type=int, default=1)
    parser.add_argument("--num_eval_envs", type=int, default=1)
    parser.add_argument("--num_env_splits", type=int, default=1)

    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=100e6,
        help='Number of environment steps to train (default: 10e6)')

    # replay buffer parameters
    parser.add_argument("--episode_length",
                        type=int,
                        default=200,
                        help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_true')

    # recurrent parameters
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr",
                        type=float,
                        default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr",
                        type=float,
                        default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--discriminator_lr", type=float, default=1e-5)
    parser.add_argument("--lagrangian_lr", type=float, default=0.1)
    parser.add_argument("--opti_eps",
                        type=float,
                        default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--algo", type=str, default="mappo")
    parser.add_argument("--ppo_epoch",
                        type=int,
                        default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument(
        "--use_clipped_value_loss",
        action='store_false',
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param",
                        type=float,
                        default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch",
                        type=int,
                        default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef",
                        type=float,
                        default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef",
                        type=float,
                        default=1,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        "--use_max_grad_norm",
        action='store_false',
        default=True,
        help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--gamma",
                        type=float,
                        default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda",
                        type=float,
                        default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        "--use_huber_loss",
        action='store_false',
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument(
        "--use_value_active_masks",
        action='store_false',
        default=True,
        help="by default True, whether to mask useless data in value loss.")
    parser.add_argument(
        "--use_policy_active_masks",
        action='store_false',
        default=True,
        help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta",
                        type=float,
                        default=10.0,
                        help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice models saving.")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="time duration between contiunous twice log printing.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.")

    # eval parameters
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="time duration between contiunous twice evaluation progress.")

    # eval & render parameters
    parser.add_argument("--eval", action='store_true', default=False)

    parser.add_argument("--render", action='store_true', default=False)
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--render_idle_time", type=float, default=0.0)
    parser.add_argument(
        "--save_video",
        action='store_true',
        default=False,
        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--video_file", type=str, default='./output.mp4')
    parser.add_argument("--video_fps", type=float, default=24)

    return parser


ALL_CONFIGS = {
    # football
    "3v1": "football/football_3v1.yaml",
    "ca_easy": "football/football_ca_easy.yaml",
    "corner": "football/football_corner.yaml",
    # smac
    "2m1z": "smac/smac_2m1z.yaml",
    "2c64zg": "smac/smac_2c64zg.yaml",
}


def make_config(type_: str, algo: str) -> Dict:
    with open(os.path.join("configs", algo, ALL_CONFIGS[type_]), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config