import torch
import numpy as np
import os
import pickle
import argparse
import json
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy, SpeedPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # speed parameters
    speed = args['speed']
    use_speed = args['use_speed_var']
    train_speed_model = args['speed_model']
    use_adjusted_speed = args['adjust_speed']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,

                         # dev
                         'speed': speed,
                         'use_speed': use_speed,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,

        # dev
        'speed': speed,
        'use_speed': use_speed,
        'speed_model': train_speed_model,
        'adjust_speed': use_adjusted_speed,
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            # success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=False, latency=0)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()

        # save results
        results_path = os.path.join(ckpt_dir, f'results_best.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)

        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, use_speed=use_speed)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    config_path = os.path.join(ckpt_dir, f'recent_model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)
    args_path = os.path.join(ckpt_dir, f'recent_model_args.json')
    with open(args_path, 'w') as f:
        json.dump(args, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    print("Making policy:", policy_class, policy_config)
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True, latency=0):
    print(f'evaluating {ckpt_name}...')
    # print('Latency: ', latency)

    set_seed(config['seed'])
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # dev
    speed = config['speed'] if 'speed' in config else 1.0
    use_speed = config['use_speed'] if 'use_speed' in config else False
    speed_model = config['speed_model'] if 'speed_model' in config else False
    adjust_speed = config['adjust_speed'] if 'adjust_speed' in config else False

    assert not (not use_speed and speed_model)
    assert not (not speed_model and adjust_speed)

    print('Using speed: ', speed)
    if adjust_speed:
        # speed_momentum = 0.99  # not used
        target_speed_momentem = 0.9
        speed_p = 0.03
        speed_d = 0.9
        print('Adjusting speed')

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    policy = make_policy(policy_class, policy_config)
    print("Loading policy...")

    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    # dev: load speed model
    if speed_model:

        config_speed = {
            'num_epochs': config['num_epochs'],
            'ckpt_dir': config['ckpt_dir'],
            'episode_len': config['episode_len'],
            'lr': config['lr'],
            'task_name': task_name,
            'seed': config['seed'],
            'camera_names': camera_names,
            'weight_decay': 1e-4,
            'backbone': 'resnet18',
            'hidden_dim': 256,
            'position_embedding': 'sine',
            'dilation': False,
            'lr_backbone': 1e-5,
            'masks': False,
        }

        speed_model_path = os.path.join(ckpt_dir, 'speed_model/policy_best.ckpt')
        speed_model = SpeedPolicy(config_speed)
        loading_status = speed_model.load_state_dict(torch.load(speed_model_path))
        print(loading_status)
        speed_model.cuda()
        speed_model.eval()
        print(f'Loaded: {speed_model_path}')



    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 1

    # dev: speed
    if use_speed:
        record_speed_res = True

        speed_curriculum = [0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0]
        #speed_curriculum = [0.6, 0.8, 1.05, 1.15, 1.75, 2.25, 2.5, 2.75]
        #speed_curriculum = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
        #speed_curriculum = [0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        #speed_curriculum = [0.1, 0.8, 4.0]

        speed_res_dict = {}
        for speed_item in speed_curriculum:
            speed_res_dict[repr(speed_item)] = [0, 0]

        speed_schedule = np.repeat(speed_curriculum, num_rollouts)
        num_rollouts = len(speed_schedule)

    if speed_model:
        image_sequence = []
        qpos_sequence = []
        speed_sample_segment = 22
        num_images = 3

    episode_returns = []
    highest_rewards = []
    for rollout_id in tqdm(range(num_rollouts)):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        # dev: speed
        if use_speed:
            speed = speed_schedule[rollout_id]
            org_speed = speed
        print('Using speed: ', speed)

        if not adjust_speed:
            cur_max_timesteps = max(int(max_timesteps / speed + 1e-8) + 1, max_timesteps // 2 +1)
        else:
            cur_max_timesteps = min(max(int(max_timesteps / speed + 1e-8) + 1, max_timesteps +1), 2 * max_timesteps)
      
        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([cur_max_timesteps, cur_max_timesteps+num_queries, state_dim]).cuda()

        #qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []

        if speed_model:
            image_sequence = []
            qpos_sequence = []

        if adjust_speed:
            error = None
            target_speed = None

            speed_history = []
            target_speed_history = []
            cur_pred_history = []

        with torch.inference_mode():
            for t in range(cur_max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

                #qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                if speed_model:
                    qpos_sequence.append(qpos)
                    image_sequence.append(curr_image)


                ### dev: speed_model
                if speed_model:
                    if len(image_sequence) < speed_sample_segment \
                        or len(qpos_sequence) < speed_sample_segment:
                        pass
                    else:
                        image_slice = image_sequence[-speed_sample_segment: :10]
                        qpos_slice = qpos_sequence[-speed_sample_segment: :10]

                        image_slice = torch.stack(image_slice, dim=2)
                        qpos_slice = torch.stack(qpos_slice, dim=1)

                        speed_pred = speed_model(image_slice.cuda(), qpos_slice.cuda(), None)

                        if t % 10== 0:
                            print('Predicted Speed: ', speed_pred.item())

                        if adjust_speed:
                            speed_history.append(speed)

                            avg_speed = np.mean(speed_history[-speed_sample_segment:])

                            cur_speed = min(avg_speed / speed_pred.item(), 5.0)

                            if target_speed is None:
                                target_speed = cur_speed
                            else:
                                target_speed = target_speed_momentem * target_speed + (1 - target_speed_momentem) * cur_speed

                            target_speed_history.append(target_speed)
                            cur_pred_history.append(cur_speed)

                            # speed = speed_momentum * speed + (1 - speed_momentum) * cur_speed

                            prev_error = error
                            error = target_speed - speed

                            # p term
                            speed = speed + speed_p * error

                            # d term
                            if prev_error is not None:
                                speed = speed + speed_d * (error - prev_error)
                            
                            print('Adjusting speed to: {:.4f} | target speed: {:.4f} | cur aim: {:.4f}'.format(speed, target_speed, cur_speed))

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image, speed=torch.Tensor([[speed]]).cuda())
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]

                        ### introduce latency
                        if latency != 0:
                            if len(actions_for_curr_step) <= latency:
                                # only keep the first action
                                actions_for_curr_step = actions_for_curr_step[:1]
                            else:
                                # delete last actions
                                actions_for_curr_step = actions_for_curr_step[:-latency]

                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if use_speed and record_speed_res:
            if env_max_reward == episode_highest_reward:
                speed_res_dict[repr(org_speed)][0] += 1
            speed_res_dict[repr(org_speed)][1] += 1

            # save to json
            with open(os.path.join(ckpt_dir, 'speed_res.json'), 'w') as f:
                json.dump(speed_res_dict, f, indent=4)

        if save_episode:
            if not adjust_speed:
                save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}_v{speed}.mp4'))
            else:
                save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}_v{org_speed}_adjusted.mp4'))

                # plot speed
                plt.figure()
                plt.plot(speed_history, label='speed', color='blue')
                plt.plot(target_speed_history, label='target speed', color='red')
                plt.plot(cur_pred_history, label='ideal speed by prediction', color='green')
                plt.xlabel('time step')
                plt.ylabel('speed')
                plt.title(f'Adjustment for initial speed = {org_speed}')
                plt.axhline(y=1, color='black', linestyle='--', label='env speed')
                plt.legend()
                plt.savefig(os.path.join(ckpt_dir, f'speed_v{org_speed}_adjusted.png'))
                plt.close()


    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, speed = data
    image_data, qpos_data, action_data, is_pad, speed = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), speed.cuda()
    return policy(qpos_data, image_data, action_data, is_pad, speed) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config, train_speed_model=False):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # dev feats
    parser.add_argument('--speed', action='store', type=float, help='speed', default=1.0)
    parser.add_argument('--use_speed_var', action='store_true', default=False)
    parser.add_argument('--speed_model', action='store_true', default=False)
    parser.add_argument('--adjust_speed', action='store_true', default=False)


    main(vars(parser.parse_args()))
