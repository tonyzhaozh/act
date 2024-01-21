import time
import numpy as np
from policy_speed_env import create_speed_env
#from rl.agents.DQN import DQNAgent
from rl.rainbowDQN.dqnAgent import DQNAgent
from utils import set_seed

from ee_sim_env import DISABLE_RENDER, DISABLE_RANDOM

import argparse

import IPython
e = IPython.embed

def main(args):
    mode = 'learned' if args['learned_policy'] else 'scripted'
    run(mode, args, is_eval=args['eval'])

def run(mode='scripted', args=None, is_eval = False):
    ######################################################

    CONSTANT_SPEED_SLOT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_evals = 50
    
    seed = args['seed']

    # training
    num_frames = 2000000
    batch_size = 256

    # RL params
    frame_skip = 10
    gamma = 0.99
    memory_size = 2000000
    target_update = 50  # TODO tony
    num_frames = num_frames // frame_skip
    hidden_dim = 1024  # TODO tony
    lr = args['lr']

    # env params
    speed_param = (0.5, 0.25, 10)  # min_speed, speed_slot_val, slot_num
    high_speed = speed_param[0] + speed_param[1] * (speed_param[2] - 1)

    CONSTANT_SPEED_LIST = [speed_param[0] + speed_param[1] * ii for ii in CONSTANT_SPEED_SLOT_LIST]

    # pow 2 lower weight
    reward_name = "Pow2Low"
    def reward_fn(speed, done, success):
        reward = 80 if done and success else 0
        reward += (speed ** 2.0) / 120
        return reward

    is_sim = True
    if args and args['task_name'][:4] != 'sim_' and args['task_name'] != 'multitask':
        is_sim = False

    # saving
    ckpt_save_freq = 10000 if is_sim else 500

    # others
    disable_render = not is_eval and not args["onscreen_render"]
    disable_random = False
    disable_env_state = False
    load_model = True and not args["new"]
    num_tests = 1

    # name
    name = f"{mode}Policy_SpeedReward{reward_name}_fs{frame_skip}"
    if not disable_random:
        print("Using random")
        name += "_Rand005"
    if speed_param != (1.0, 0.5, 5):
        name += f"_vmax{high_speed}-{speed_param[2]}"
    if disable_env_state:
        name += "_no_envs"
    if hidden_dim != 256:
        name += f"_dim{hidden_dim}"

    if args is not None:
        name = args["task_name"] + '_' + name
        is_sim = args["task_name"][:4] == "sim_"
    name += f"_seed{seed}"

    model_path = f"/scr2/tonyzhao/dynamic_train_logs/{name}"

    ######################################################

    if not is_eval:
        set_seed(seed)

    DISABLE_RENDER[0] = disable_render
    DISABLE_RANDOM[0] = disable_random
    if mode == 'scripted':
        env = create_speed_env( mode, None, task_name=args['task_name'],
            reward_fn=reward_fn, use_parallel=False,
            speed_param = speed_param, use_env_state= not disable_env_state,
            onscreen_render= is_eval or args["onscreen_render"]
        )
    elif mode == 'learned':
        env = create_speed_env( mode, args, task_name=args['task_name'],
            reward_fn=reward_fn, use_parallel=False,
            speed_param = speed_param, use_env_state= not disable_env_state,
            onscreen_render = is_eval, save_video=is_eval
        )
    else:
        raise NotImplementedError('Unrecognized mode')
    
    speed_success_list = []
    for constant_speed in CONSTANT_SPEED_SLOT_LIST:
        print(f'{constant_speed=}')
        success_list = []
        for i in range(num_evals):
            state, obs = env.reset()
            episode_return = 0
            episode_action = []
            done = False
            while not done:
                action = constant_speed
                next_state, next_obs, reward, done, info = env.step(action, frame_skip)
                #print(next_state, reward, done, info, self.frame_skip)

                success = info['success']
                force_finish = info['finish']

                state = next_state
                episode_return += reward
                episode_action.append(action)
            success_list.append(success)
        speed_success_list.append(np.mean(success_list))

    print('\n\n\n')
    print(f'{CONSTANT_SPEED_SLOT_LIST=}')
    print(f'{CONSTANT_SPEED_LIST=}')
    print(f'{speed_success_list=}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--learned-policy', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # eval
    parser.add_argument('--eval', action='store_true')

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)

    # dummy 
    parser.add_argument('--policy_class', action='store')
    parser.add_argument('--num_epochs', action='store')

    main(vars(parser.parse_args()))

