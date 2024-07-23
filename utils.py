import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.transform import Rotation
import glob
import random
import io
from PIL import Image

def euler_to_r6(euler, degrees=False):
    rot_mat = Rotation.from_euler("xyz", euler, degrees=degrees).as_matrix()
    a1, a2 = rot_mat[0], rot_mat[1]
    return np.concatenate((a1, a2)).astype(np.float32)

import IPython
e = IPython.embed

def get_qpos(root):
    # goes together with the function below! Do not change separately!
    xyz = root['observations']['robot_poses'][:, :3]  # get the xyz
    euler = root['observations']['robot_poses'][:, 3:]  # get the euler angles and convert to r6
    r6s = np.array([euler_to_r6(degrees, degrees=True) for degrees in euler])
    return np.concatenate([xyz, r6s], axis=1)

def get_single_qpos(pose):
    # goes together with the function above! Do not change separately!
    xyz = pose[:3]
    joint_angles = pose[3:]
    r6s = euler_to_r6(joint_angles, degrees=True)
    return np.concatenate([xyz, r6s], axis=0)
    

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_filenames, dataset_dir, camera_names, norm_stats, slice_episode_len, zero_qpos=False):
        super(EpisodicDataset).__init__()
        self.episode_filenames = episode_filenames
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.slice_episode_len = slice_episode_len
        self.zero_qpos = zero_qpos
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_filenames)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode
        # print('getting index', index)

        dataset_path = self.episode_filenames[index]
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            episode_len, action_dim = root['/action'].shape
            
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            # get observation at start_ts only
            qpos = get_qpos(root)[start_ts]

            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros([self.slice_episode_len, action_dim], dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.slice_episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            image = image_dict[cam_name]
            if isinstance(image, np.bytes_): # if image is compressed JPEG string
                jpg_bytes = io.BytesIO(image)
                image = Image.open(jpg_bytes)
                image = np.array(image)
                all_cam_images.append(image)
            else:
                all_cam_images.append(image)
                
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        if self.zero_qpos:
            qpos_data = torch.zeros_like(qpos_data)
        else:
            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # print('image_data', image_data.shape)
        # print('qpos_data', qpos_data.shape)
        # print('action_data', action_data.shape)
        # print('is_pad', is_pad.shape)

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(all_hdf5_files):
    all_qpos_data = []
    all_action_data = []
    episode_lens = []
    for file in all_hdf5_files:
        # print("opening ", file)
        try:
            with h5py.File(file, 'r') as root:
                qpos = get_qpos(root)
                action = root['/action'][()]
        except:
            print("error opening ", file)
            continue
        all_qpos_data.append(torch.from_numpy(qpos))
        episode_lens.append(action.shape[0])
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}    

    return stats, np.max(np.array(episode_lens))


def load_data(dataset_dir, max_num_episodes, camera_names, batch_size_train, batch_size_val, task_config, zero_qpos=False):
    print(f'\nData from: {dataset_dir}\n')

    if 'stage_key' in task_config:  # for multi stage tasks
        stage_key = task_config['stage_key']
        search_pattern = f'**/*/*{stage_key}.hdf5'
    else:
        search_pattern = '**/*/*.hdf5'

    if isinstance(dataset_dir, list):
        all_hdf5 = []
        for d in dataset_dir:
            new_files = glob.glob(os.path.join(d, search_pattern), recursive=True)
            assert len(new_files) > 0, f"no hdf5 files found in {d}"
            all_hdf5 += new_files
    else:
        all_hdf5 = glob.glob(os.path.join(dataset_dir, search_pattern), recursive=True)
    
    print(f"found {len(all_hdf5)} hdf5 files")
    random.shuffle(all_hdf5)

    if max_num_episodes == -1:
        num_episodes = len(all_hdf5)
    else:
        num_episodes = min(num_episodes, len(all_hdf5))

    # obtain train test split
    train_ratio = 0.9
    train_files = all_hdf5[:int(train_ratio * num_episodes)]
    val_files = all_hdf5[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats, max_episode_len = get_norm_stats(all_hdf5)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_files, dataset_dir, camera_names, norm_stats, max_episode_len, zero_qpos=zero_qpos)
    val_dataset = EpisodicDataset(val_files, dataset_dir, camera_names, norm_stats, max_episode_len, zero_qpos=zero_qpos)
    print('train_dataset len:', len(train_dataset))
    print('val_dataset len:', len(val_dataset))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
