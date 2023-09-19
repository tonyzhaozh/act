import numpy as np
import torch
import os
import json
import random
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = Trueode_id = self.episode_ids[index]
            dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                is_sim = root.attrs['sim']
                original_action_shape = root['/action'].shape
                episode_len = original_action_shape[0]
                if sample_full_episode:
                    start_ts = 0
                else:
                    start_ts = np.random.choice(episode_len)
                # get observation at start_ts only
                qpos = root['/observations/qpos'][start_ts]
                qvel = root['/observations/qvel'][start_ts]
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
            padded_action = np.zeros(original_action_shape, dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(episode_len)
            is_pad[action_len:] = 1

            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
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
            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

            return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, 10) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, 10) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
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
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)


def interpolate_by_step(raw, step, max_length = None, with_batch = True):
    # raw: numpy array of shape (N, 14)
    # step: a positive real number signifying the step length
    # interpolate the raw data of shape (14, )
    if step < 1e-8:
        raise ValueError('Invalid step size')

    if with_batch:
        assert raw.shape[0] == 1
        raw = raw.squeeze(0)

    if isinstance(raw, list):
        raw = torch.tensor(raw)
    elif isinstance(raw, np.ndarray):
        raw = torch.from_numpy(raw)
    elif isinstance(raw, torch.Tensor):
        raw = raw.cpu()  # Move tensor from GPU to CPU

    n = raw.shape[0]  # Number of rows in raw data

    out = []
    cnt = 0.0
    while cnt < n - 1 + 1e-8:
        start_idx = int(cnt)
        end_idx = min(start_idx + 1, n - 1)

        start_point = raw[start_idx]
        end_point = raw[end_idx]

        fraction = cnt - start_idx
        interpolated_point = start_point + fraction * (end_point - start_point)

        out.append(interpolated_point)
        cnt += step

        if max_length != None and len(out) >= max_length:
            break

    out_tensor = torch.stack(out)  # Stack the list of tensors into a single tensor

    if isinstance(raw, torch.Tensor):
        out_tensor = out_tensor.cuda()  # Move output back to the same device
    
    if with_batch:
        out_tensor = out_tensor.unsqueeze(0)
    
    return out_tensor


def sample_speed(minVal = 0.1, maxVal = 10.0):
    rand_num = np.random.uniform(0, 1)

    if rand_num < 0.5:
        return np.random.uniform(minVal, 1)
    else:
        return np.random.uniform(1, maxVal)

def append_results(entry, filename):
    #  Check if the file exists
    if not os.path.exists(filename):
        # File does not exist, create it with an empty list
        with open(filename, 'w') as file:
            json.dump([], file)

    # Read the existing data from the file
    try:
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    except:
        with open(filename, 'w') as file:
            json.dump([], file)
        existing_data = []

    # Append the new entry to the existing data
    existing_data.append(entry)

    # Write the updated data back to the file
    with open(filename, 'w') as file:
        json.dump(existing_data, file)


# testing purpose
if __name__=="__main__":
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def number_to_one_hot(number, size=501):
    one_hot_array = np.zeros(size)
    one_hot_array[number] = 1
    return one_hot_array
