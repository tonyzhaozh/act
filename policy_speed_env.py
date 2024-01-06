import random

import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from ee_sim_env import make_ee_sim_env
from scripted_policy import PickAndTransferTeaBagPolicy, PickAndTransferPolicy, InsertionPolicy
from utils import number_to_one_hot
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim

import cv2, copy

from act_policy_wrapper import InterpolatedACTPolicy
from utils import sample_teabag_pose, get_task_config
from sim_env import BOX_POSE

from constants import PUPPET_GRIPPER_JOINT_OPEN, SIM_TASK_CONFIGS

import IPython
e = IPython.embed

class SpeedPolicyEnv:

    def __init__(self, env, policy, reward_fn, episode_len, is_sim,
                 save_video=False, onscreen_render=True, use_state=True, use_env_state=True, use_obs = False,
                 parallel_env = None, parallel_policy = None,
                 speed_param=(1.0, 0.5, 5), env_pre_reset_script = None, env_finish_script = None, multitask = False
        ):
        self.env = env
        self.is_sim = is_sim
        self.env_pre_reset_script = env_pre_reset_script
        self.env_finish_script = env_finish_script
        self.policy = policy
        self.reward_fn = reward_fn
        self.episode_len = episode_len
        self.save_video = save_video
        self.onscreen_render = onscreen_render
        self.multitask = multitask

        self.timestep_cnt = 0
        self.real_cnt = 0
        self.episode = []
        self.cur_ts = None
        self.cur_success = False
        self.success_keyframe = None
        self.success_keyframe_real = None


        self.min_speed, self.speed_slot_val, self.speed_slot_num = speed_param
        self.speed_list = []

        self.use_state = use_state
        self.use_env_state = use_env_state
        self.use_obs = use_obs
        if use_state:
            if self.use_env_state:
                self.obs_space = 39 + 14 + 14
            else:
                self.obs_space = 14 + 14
        else:
            self.obs_space = 96 * 128 * 3 + 14
        self.action_space = self.speed_slot_num

        if self.onscreen_render:
            self.ax = None
            self.plt_img = None

        if self.save_video:
            self.image_list = []

        if parallel_env is not None:
            assert parallel_policy is not None
            self.use_parallel_env = True
            self.parallel_env = parallel_env
            self.parallel_policy = parallel_policy
            self.parallel_cur_ts = None
            self.difference_list = []
            self.difference_list_mse = []
        else:
            self.use_parallel_env = False

        if self.multitask:
            assert isinstance(env, list)
            assert isinstance(policy, list)
            assert isinstance(episode_len, list)
            assert not self.use_parallel_env
            self.envs = env
            self.policies = policy
            self.episode_lens = episode_len
            self.policy = None
            self.env = None
            self.episode_len = None
            self.task_index = None
            self.task_num = len(self.envs)
            self.obs_space += self.task_num


    def reset(self):
        if self.env_pre_reset_script is not None:
            self.env_pre_reset_script()

        if self.env is not None:
            self.cur_ts = self.env.reset()
        if self.policy is not None:
            self.policy.reset()
        self.timestep_cnt = 0
        self.real_cnt = 0
        self.image_list = []
        self.episode = []
        self.cur_success = False
        self.success_keyframe = None
        self.success_keyframe_real = None

        if self.use_parallel_env:
            self.parallel_cur_ts = self.parallel_env.reset()
            self.parallel_policy.reset()
            self.difference_list = []
            self.difference_list_mse = []

        if self.multitask:
            self.task_index = random.randint(0, self.task_num - 1)
            self.env = self.envs[self.task_index]
            self.policy = self.policies[self.task_index]
            self.episode_len = self.episode_lens[self.task_index]
            self.cur_ts = self.env.reset()
            self.policy.reset()

        if self.onscreen_render:
            self.ax = plt.subplot()
            self.plt_img = self.ax.imshow(self.cur_ts.observation['images']['angle'])
            plt.ion()


        if not self.is_sim:
            input("Ready to go.")

        return self.get_obs()
        #return number_to_one_hot(0)

    def step(self, speed, quantized = True):
        if self.timestep_cnt >= self.episode_len:
            raise ValueError("Cannot proceed further: the task is done")

        if quantized:
            if isinstance(speed, float):
                raise ValueError("Speed is set to be quantized, invalid value detected")
            speed = self.min_speed + self.speed_slot_val * min(speed, self.speed_slot_num)

        self.timestep_cnt += speed
        self.real_cnt += 1

        self.speed_list.append(speed)

        if self.use_parallel_env:
            parallel_speed = int(speed)
            self.parallel_policy.step_count = self.policy.step_count

            while parallel_speed > 0:
                if self.parallel_policy.step_count >= self.episode_len:
                    break
                action = self.parallel_policy(self.parallel_cur_ts)
                try:
                    self.parallel_cur_ts = self.parallel_env.step(action)
                except Exception as ee:
                    print("Warning: bad physics")

                parallel_speed -= 1

            #image_parallel_difference = self.get_parallel_difference()
            #self.difference_list.append(image_parallel_difference)

            if not self.use_state:
                image_parallel_difference_mse = self.get_parallel_difference_mse()
                self.difference_list_mse.append(image_parallel_difference_mse)
            else:
                state_parallel_difference = self.get_parallel_difference_state()

        #print(self.cur_ts)
        action = self.policy(self.cur_ts, step_inc=speed)

        try:
            self.cur_ts = self.env.step(action)
        except Exception as e:
            print("Warning: bad physics", e)
            state, observation = self.get_obs()
            return state, observation, 0.0, True, {'success': False, 'finish': False}

        self.episode.append(self.cur_ts)


        if self.onscreen_render:
            self.plt_img.set_data(self.cur_ts.observation['images']['angle'])
            plt.pause(0.01)

        if self.save_video:
            self.image_list.append(self.cur_ts.observation['images']['angle'])

        state, observation = self.get_obs()
        # print(observation.shape)

        done = self.timestep_cnt >= self.episode_len

        reward = self.cur_ts.reward
        force_finish = False
        if done and not self.is_sim:
            reward, force_finish = self.user_reward_interface()

        if reward >= 100:
            self.cur_success = True
            self.success_keyframe = self.timestep_cnt
            self.success_keyframe_real = self.real_cnt

        # done = reward == 100.0 or self.timestep_cnt >= self.episode_len

        if self.reward_fn is not None:
            ret_reward = self.reward_fn(speed, done, self.cur_success)
        else:
            ret_reward = reward

        # if self.use_parallel_env:
        #     if not self.use_state:
        #         #ret_reward += (20 - image_parallel_difference_mse) / 20 / 10
        #         pass
        #     else:
        #         ret_reward -= state_parallal_difference

        # print("Step:", self.timestep_cnt, " Speed:", speed, " Reward:", reward)

        if done:
            if self.multitask:
                print(f"Multitask Index: {self.task_index}")
            print(f"Episode Done. Success:{self.cur_success} Keyframe:{self.success_keyframe} Real:{self.success_keyframe_real}")

            #print("Step counts:", self.real_cnt, self.timestep_cnt)
            # plt.close()
            # plt.plot(self.speed_list)
            # plt.savefig("tmp/speed.png")

            if self.env_finish_script is not None:
                self.env_finish_script()

            if self.use_parallel_env:
                #plt.close()
                #plt.plot(self.difference_list)
                #plt.savefig("tmp/diff.png")

                #plt.close()
                #plt.plot(self.difference_list_mse)
                #plt.savefig("tmp/diff_mse.png")

                pass

            if self.save_video:
                output_file = 'visualization/test_output_video.mp4'
                codec = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 50

                # Assuming images is your list of numpy arrays representing images
                height, width, channels = self.image_list[0].shape

                video_writer = cv2.VideoWriter(output_file, codec, fps, (width, height))

                for image in self.image_list:
                    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                    video_writer.write(bgr_image)

                video_writer.release()

                print('saved video')


        #return observation, reward, done, {}  # for policies conditioned on images
        return state, observation, ret_reward, done, {'success': self.cur_success, 'finish': force_finish}  # for policies conditioned on images
        #return number_to_one_hot(min(500, int(self.timestep_cnt))), reward, done, {}  # for policies conditioned on time

    def get_obs(self):
        parallel = self.use_parallel_env
        use_state = self.use_state
        use_env = self.use_env_state
        use_obs = self.use_obs
        assert not (parallel and use_state)
        assert not parallel

        if use_state:
            if use_env:
                env_state = self.cur_ts.observation['env_state']
                out = np.concatenate([
                    env_state,
                    self.cur_ts.observation["qpos"],
                    np.array(self.cur_ts.observation["qvel"], dtype=float)
                ])
            else:
                out = np.concatenate([
                    self.cur_ts.observation["qpos"],
                    np.array(self.cur_ts.observation["qvel"], dtype=float)
                ])

            if self.multitask:
                out = np.concatenate([out, np.eye(self.task_num)[self.task_index]], dtype=float)
        else:
            raise NotImplementedError

        # if not parallel:
        #     image = self.cur_ts.observation['images']['angle']
        # else:
        #     assert self.use_parallel_env
        #     image = self.parallel_cur_ts.observation['images']['angle']

        image = None
        if use_obs:
            image = self.cur_ts.observation['images']['angle']
            image = image / 255.0 - 0.5
        return out, image

    def get_parallel_difference(self):
        img1 = self.cur_ts.observation['images']['angle']
        img2 = self.parallel_cur_ts.observation['images']['angle']

        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same shape")

        # Compute SSIM for each channel (R, G, B)
        channel_ssims = []
        for channel in range(3):  # 0: Red, 1: Green, 2: Blue
            channel_ssim = ssim(img1[:, :, channel], img2[:, :, channel], data_range=img1.max() - img1.min())
            channel_ssims.append(channel_ssim)

        # Calculate the mean SSIM over all channels
        mean_ssim = np.mean(channel_ssims)

        return mean_ssim

    def get_parallel_difference_mse(self):
        img1 = self.cur_ts.observation['images']['angle']
        img2 = self.parallel_cur_ts.observation['images']['angle']

        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same shape")

        squared_diff = np.square(img1 - img2)
        mse = np.mean(squared_diff)

        return mse

    def get_parallel_difference_state(self):
        env1 = self.cur_ts.observation['env_state']
        env2 = self.parallel_cur_ts.observation['env_state']

        qpos1 = self.cur_ts.observation["qpos"]
        qpos2 = self.parallel_cur_ts.observation["qpos"]

        env_diff = np.mean(np.square(env1 - env2))
        qpos_diff = np.mean(np.square(qpos1 - qpos2))

        #print("Env diff:", env_diff)
        #print("Qpos diff:", qpos_diff)

        return env_diff + qpos_diff

    def user_reward_interface(self):
        while True:
            try:
                ret = input("\nEpisode finished. Success: ").split(" ")
                raw_reward = float(ret[0])
                assert -1 <= raw_reward <= 1
                force_exit = False
                if len(ret) >= 2 and ret[1] == 'exit':
                    force_exit = True
                break
            except Exception as e:
                print("Input Error: must be in the format \"[reward(float)] [options]\"")
        return raw_reward * 100, force_exit

    def close(self):
        plt.close()


def test_speed_env(task_name = 'sim_transfer_tea_bag_scripted', speed_func=None, speed_func_generator=None,
                   save_video = False, onscreen_render=False, use_parallel = False,
                    speed_param = (1.0, 0.5, 5),
    ):

    if 'sim_transfer_tea_bag' in task_name:
        env = make_ee_sim_env('sim_transfer_tea_bag')
        policy = PickAndTransferTeaBagPolicy()
        episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    elif 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
        policy = PickAndTransferPolicy()
        episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
        policy = InsertionPolicy()
        episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    elif task_name == 'multitask':
        task_names = ['sim_transfer_tea_bag', 'sim_transfer_cube', 'sim_insertion']
        env = [make_ee_sim_env(task_name) for task_name in task_names]
        policy = [PickAndTransferTeaBagPolicy(), PickAndTransferPolicy(), InsertionPolicy()]
        episode_len = [SIM_TASK_CONFIGS[task_name + '_scripted']['episode_len'] for task_name in task_names]
    else:
        raise NotImplementedError

    if not use_parallel:
        speed_env = SpeedPolicyEnv(env, policy, None, episode_len, True, use_state=False, onscreen_render=onscreen_render, save_video=save_video, speed_param=speed_param, multitask= task_name == 'multitask')
    else:
        speed_env = SpeedPolicyEnv(env, policy, None, episode_len, True, use_state=False, onscreen_render=onscreen_render, speed_param=speed_param,
                                   parallel_env=copy.deepcopy(env), parallel_policy=copy.deepcopy(policy), save_video=save_video)

    assert not (speed_func and speed_func_generator)
    if speed_func_generator is not None:
        speed_func = speed_func_generator(episode_len)

    num_rollouts = 3
    for rollout in range(num_rollouts):
        state, obs = speed_env.reset()
        done = False
        rewards = []
        while not done:
            if speed_func is None:
                #speed = np.random.uniform(low=0.1, high=10.0)
                speed = 3.0
            else:
                speed = speed_func(obs=obs, t=speed_env.timestep_cnt)

            obs, reward, done, _ = speed_env.step(speed, quantized=False)
            rewards.append(reward)

        max_reward = np.max(rewards)
        print("Max Reward:", max_reward)

    speed_env.close()
    return max_reward

def test_act_speed_env(task_name = 'sim_transfer_tea_bag_scripted', speed_func=None, speed_func_generator=None,
                   save_video = False, onscreen_render=False, use_parallel = False,
                    speed_param = (1.0, 0.5, 5),
    ):
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len'] * 1.1
    if 'sim_transfer_tea_bag' in task_name:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

        def resample_box_pos():
            BOX_POSE[0] = sample_teabag_pose()
    else:
        raise NotImplementedError

    args = {
        'task_name': 'sim_transfer_tea_bag_scripted',
        'ckpt_dir': '/scr2/tonyzhao/train_logs/sim_transfer_tea_bag_scripted',
        'lr': 1e-5,
        'chunk_size': 100,
        'kl_weight': 80,
        'hidden_dim': 512,
        'dim_feedforward': 3000
    }

    policy = InterpolatedACTPolicy(args)

    speed_env = SpeedPolicyEnv(env, policy, None, episode_len, use_state=False, onscreen_render=onscreen_render, save_video=save_video,
                               speed_param=speed_param, env_pre_reset_script=resample_box_pos)

    assert not (speed_func and speed_func_generator)
    if speed_func_generator is not None:
        speed_func = speed_func_generator(episode_len)

    state, obs = speed_env.reset()
    done = False
    rewards = []
    while not done:
        if speed_func is None:
            #speed = np.random.uniform(low=0.1, high=10.0)
            speed = 0.25
        else:
            speed = speed_func(obs=obs, t=speed_env.timestep_cnt)

        state, obs, reward, done, _ = speed_env.step(speed, quantized=False)
        rewards.append(reward)

    max_reward = np.max(rewards)
    print("Max Reward:", max_reward)
    speed_env.close()

    return max_reward

def create_speed_env(
        mode = "scripted", args = None,
        task_name = 'sim_transfer_tea_bag_scripted', reward_fn = None, onscreen_render=False, use_parallel = False, use_env_state = True, save_video=False,
        speed_param = (1.0, 0.5, 5)
    ):
    print(f"Creating {mode} speed env for task {task_name} (args = {args})")

    env_pre_reset_script = None
    env_finish_script = None

    is_sim = task_name[:4] == 'sim_'

    if mode == 'scripted':
        if 'sim_transfer_tea_bag' in task_name:
            env = make_ee_sim_env('sim_transfer_tea_bag')
            policy = PickAndTransferTeaBagPolicy()
            task_config = get_task_config(task_name)
            episode_len = task_config['episode_len']
        elif 'sim_transfer_cube' in task_name:
            env = make_ee_sim_env('sim_transfer_cube')
            policy = PickAndTransferPolicy()
            task_config = get_task_config(task_name)
            episode_len = task_config['episode_len']
        elif 'sim_insertion' in task_name:
            env = make_ee_sim_env('sim_insertion')
            policy = InsertionPolicy()
            task_config = get_task_config(task_name)
            episode_len = task_config['episode_len']
        elif task_name == 'multitask':
            is_sim  = True
            task_names = ['sim_transfer_tea_bag', 'sim_transfer_cube', 'sim_insertion']
            env = [make_ee_sim_env(task_name) for task_name in task_names]
            policy = [PickAndTransferTeaBagPolicy(), PickAndTransferPolicy(), InsertionPolicy()]
            episode_len = [get_task_config(task_name + '_scripted')['episode_len'] for task_name in task_names]
        else:
            raise NotImplementedError

    elif mode == 'learned' and 'sim_transfer_tea_bag' in task_name:
        assert args is not None
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

        task_config = get_task_config(task_name)
        episode_len = task_config['episode_len']

        def resample_box_pos():
            BOX_POSE[0] = sample_teabag_pose()
        policy = InterpolatedACTPolicy(args)
        env_pre_reset_script = resample_box_pos
    elif mode == 'learned' and not is_sim:
        # load environment
        print("Running RL in real environment")

        task_config = get_task_config(task_name)
        episode_len = task_config['episode_len']

        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0

        policy = InterpolatedACTPolicy(args)
        def open_gripper():
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                          move_time=0.5)  # open
        env_finish_script = open_gripper

    else:
        raise NotImplementedError

    if not use_parallel:
        speed_env = SpeedPolicyEnv(env, policy, reward_fn, episode_len, is_sim, onscreen_render=onscreen_render, use_env_state=use_env_state,
                                   save_video=save_video, speed_param=speed_param,
                                   env_pre_reset_script = env_pre_reset_script, env_finish_script = env_finish_script, multitask= task_name == 'multitask')
    else:
        speed_env = SpeedPolicyEnv(env, policy, reward_fn, episode_len, is_sim, onscreen_render=onscreen_render, use_env_state=use_env_state,
                                   speed_param=speed_param, parallel_env = copy.deepcopy(env), parallel_policy = copy.deepcopy(policy), save_video=save_video,
                                   env_pre_reset_script=env_pre_reset_script, env_finish_script = env_finish_script)

    return speed_env

if __name__=='__main__':
    task1 = 'sim_transfer_tea_bag_scripted'
    task2 = 'sim_transfer_cube_scripted'
    task3 = 'sim_insertion_scripted'
    multitask = 'multitask'
    test_speed_env(task_name=multitask, onscreen_render=True)
    '''
    reward = test_speed_env()
    while reward == 100:
        reward = test_speed_env()
    '''
    try:
        #reward = test_speed_env()
        pass
    except:
        pass
    #reward = test_act_speed_env(onscreen_render=True, save_video=False)
