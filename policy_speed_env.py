import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from scripted_policy import PickAndTransferTeaBagPolicy
from utils import number_to_one_hot
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim

import cv2, copy

import IPython
e = IPython.embed

class SpeedPolicyEnv:

    def __init__(self, env, policy, episode_len, save_video=False, onscreen_render=True, use_state=True, parallel_env = None, parallel_policy = None):
        self.env = env
        self.policy = policy
        self.episode_len = episode_len
        self.save_video = save_video
        self.onscreen_render = onscreen_render

        self.timestep_cnt = 0
        self.real_cnt = 0
        self.episode = []
        self.cur_ts = None
        self.cur_success = False

        self.min_speed = 1.0
        self.speed_slot_val = 0.5
        self.speed_slot_num = 5
        self.speed_list = []

        self.use_state = use_state
        if use_state:
            self.obs_space = 39 + 14 + 14
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

    def reset(self):
        self.cur_ts = self.env.reset()
        self.policy.reset()
        self.timestep_cnt = 0
        self.real_cnt = 0
        self.image_list = []
        self.episode = []
        self.cur_success = False

        if self.use_parallel_env:
            self.parallel_cur_ts = self.parallel_env.reset()
            self.parallel_policy.reset()
            self.difference_list = []
            self.difference_list_mse = []

        if self.onscreen_render:
            self.ax = plt.subplot()
            self.plt_img = self.ax.imshow(self.cur_ts.observation['images']['angle'])
            plt.ion()

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
                except Exception as e:
                    print("Warning: bad physics")

                parallel_speed -= 1

            #image_parallel_difference = self.get_parallel_difference()
            #self.difference_list.append(image_parallel_difference)

            if not self.use_state:
                image_parallel_difference_mse = self.get_parallel_difference_mse()
                self.difference_list_mse.append(image_parallel_difference_mse)
            else:
                state_parallal_difference = self.get_parallel_difference_state()

        action = self.policy(self.cur_ts, step_inc=speed)
        try:
            self.cur_ts = self.env.step(action)
        except Exception as e:
            print("Warning: bad physics")
            return self.timestep_cnt, 0.0, True, {}

        self.episode.append(self.cur_ts)


        if self.onscreen_render:
            self.plt_img.set_data(self.cur_ts.observation['images']['angle'])
            plt.pause(0.02)

        if self.save_video:
            self.image_list.append(self.cur_ts.observation['images']['angle'])

        observation = self.get_obs()
        # print(observation.shape)

        reward = self.cur_ts.reward
        if reward >= 100:
            self.cur_success = True
        #done = reward == 100.0 or self.timestep_cnt >= self.episode_len

        done = self.timestep_cnt >= self.episode_len
        if done:
            # reward += 1
            pass
        ret_reward = 100 if done and self.cur_success else 0

        ret_reward += (speed ** 1.0) / 200

        if self.use_parallel_env:
            if not self.use_state:
                #ret_reward += (20 - image_parallel_difference_mse) / 20 / 10
                pass
            else:
                ret_reward -= state_parallal_difference

        #print("Step:", self.timestep_cnt, " Speed:", speed, " Reward:", reward)

        if done:
            #print("Step counts:", self.real_cnt, self.timestep_cnt)
            plt.close()
            plt.plot(self.speed_list)
            plt.savefig("tmp/speed.png")

            if self.use_parallel_env:
                #plt.close()
                #plt.plot(self.difference_list)
                #plt.savefig("tmp/diff.png")

                #plt.close()
                #plt.plot(self.difference_list_mse)
                #plt.savefig("tmp/diff_mse.png")

                pass

            if self.save_video:
                output_file = 'output_video.mp4'
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
        return observation, ret_reward, done, {}  # for policies conditioned on images
        #return number_to_one_hot(min(500, int(self.timestep_cnt))), reward, done, {}  # for policies conditioned on time

    def get_obs(self, parallel = False, use_state = True):
        assert not (parallel and use_state)

        if use_state:
            env_state = self.cur_ts.observation['env_state']
            out = np.concatenate([
                env_state, 
                self.cur_ts.observation["qpos"],
                np.array(self.cur_ts.observation["qvel"], dtype=float)
            ])
            return out

        if not parallel:
            image = self.cur_ts.observation['images']['angle']
        else:
            assert self.use_parallel_env
            image = self.parallel_cur_ts.observation['images']['angle']

        resized_image = zoom(image, (.2, .2, 1)).flatten()
        out = np.concatenate([resized_image, np.array(self.cur_ts.observation["qvel"], dtype=float)])
        return out

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

    def close(self):
        plt.close()


def test_speed_env(task_name = 'sim_transfer_tea_bag_scripted', speed_func=None, speed_func_generator=None, onscreen_render=False, use_parallel = True):
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_tea_bag' in task_name:
        env = make_ee_sim_env('sim_transfer_tea_bag')
    else:
        raise NotImplementedError

    policy = PickAndTransferTeaBagPolicy()

    if not use_parallel:
        speed_env = SpeedPolicyEnv(env, policy, episode_len, onscreen_render=onscreen_render, save_video=True)
    else:
        speed_env = SpeedPolicyEnv(env, policy, episode_len, onscreen_render=onscreen_render,
                                   parallel_env=copy.deepcopy(env), parallel_policy=copy.deepcopy(policy), save_video=True)

    assert not (speed_func and speed_func_generator)
    if speed_func_generator is not None:
        speed_func = speed_func_generator(episode_len)

    obs = speed_env.reset()
    done = False
    rewards = []
    while not done:
        if speed_func is None:
            #speed = np.random.uniform(low=0.1, high=10.0)
            speed = 1.0
        else:
            speed = speed_func(obs=obs, t=speed_env.timestep_cnt)

        obs, reward, done, _ = speed_env.step(speed, quantized=False)
        rewards.append(reward)

    max_reward = np.max(rewards)
    print("Max Reward:", max_reward)
    speed_env.close()

    return max_reward

def create_speed_env(task_name = 'sim_transfer_tea_bag_scripted', onscreen_render=False, use_parallel = False, save_video=False):
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_tea_bag' in task_name:
        env = make_ee_sim_env('sim_transfer_tea_bag')
    else:
        raise NotImplementedError

    policy = PickAndTransferTeaBagPolicy()
    if not use_parallel:
        speed_env = SpeedPolicyEnv(env, policy, episode_len, onscreen_render=onscreen_render, save_video=save_video)
    else:
        speed_env = SpeedPolicyEnv(env, policy, episode_len, onscreen_render=onscreen_render,
                                   parallel_env = copy.deepcopy(env), parallel_policy = copy.deepcopy(policy), save_video=save_video)

    return speed_env

if __name__=='__main__':
    test_speed_env()
