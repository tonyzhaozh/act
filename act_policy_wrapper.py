import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
from einops import rearrange
import cv2


from policy import ACTPolicy
from utils import interpolate_by_step, set_seed, interpolate_single

class InterpolatedACTPolicy:
    def __init__(self, args):
        set_seed(1000)

        task_name = args['task_name']
        is_sim = task_name[:4] == 'sim_'
        self.is_sim = is_sim
        if is_sim:
            from constants import SIM_TASK_CONFIGS
            task_config = SIM_TASK_CONFIGS[task_name]
        else:
            from aloha_scripts.constants import TASK_CONFIGS
            task_config = TASK_CONFIGS[task_name]
        self.episode_len = task_config['episode_len']
        self.camera_names = task_config['camera_names']

        ckpt_dir = args['ckpt_dir']
        ckpt_name = 'policy_best.ckpt'

        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': 1e-5,
                         'backbone': 'resnet18',
                         'enc_layers': 4,
                         'dec_layers': 7,
                         'nheads': 8,
                         'camera_names': self.camera_names,
                        }

        self.chunk_size = args['chunk_size']  # should not use
        #self.temporal_agg = args['temporal_agg']  # should not use
        self.state_dim = 14

        self.policy = ACTPolicy(policy_config)
        self.policy.eval()

        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()

        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        self.pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']

        self.step_count = 0
        self.real_step_count = 0
        self.prev_speed = None
        self.all_time_actions = torch.zeros([self.episode_len + self.chunk_size * 2, self.state_dim]).cuda()
        self.populated = torch.zeros([self.episode_len + self.chunk_size * 2]).cuda()


    def reset(self):
        self.step_count = 0
        self.real_step_count = 0
        self.prev_speed = None
        self.all_time_actions = torch.zeros([self.episode_len + self.chunk_size * 2, self.state_dim]).cuda()
        self.populated = torch.zeros([self.episode_len + self.chunk_size * 2]).cuda()


    def get_image(self, ts):
        curr_images = []
        for cam_name in self.camera_names:
            curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        return curr_image

    def __call__(self, ts, step_inc=1):
        t = self.step_count
        if self.is_sim and t >= self.episode_len + self.chunk_size:
            raise ValueError("Trajectory too long")

        if self.populated[int(t)] == 0 or self.populated[int(t) + 1] == 0:
            obs = ts.observation

            qpos_numpy = np.array(obs['qpos'])
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().unsqueeze(0).cuda()

            curr_image = self.get_image(ts).cuda()
            with torch.inference_mode():
                all_actions = self.policy(qpos, curr_image)

            cur_t = int(t)
            if t - 1 >= 0:
                cur_action = self.all_time_actions[int(t) - 1, :]
                starting_actions = all_actions[:3, :]
                distances = torch.norm(starting_actions - cur_action, dim=1)
                min_distance_index = torch.argmin(distances)
                cur_t = max(0, int(t) - min_distance_index)

            self.all_time_actions[cur_t: cur_t + all_actions.shape[1]] = all_actions[0]
            self.populated[cur_t: cur_t + all_actions.shape[1]] += 1


        raw_action = interpolate_single(t, self.all_time_actions)
        action = self.post_process(raw_action.detach().cpu().numpy())

        self.real_step_count += 1
        self.step_count += step_inc
        self.prev_speed = step_inc

        return action

