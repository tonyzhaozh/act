import torch, os, h5py, cv2
import numpy as np
import matplotlib.pyplot as plt


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, seed=0, use_augmentation=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.seed = seed
        self.use_augmentation = use_augmentation
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        #seed = self.seed
        #np.random.seed(seed)
        #random.seed(seed)
        #torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            #torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = True
            #torch.backends.cudnn.deterministic = True
            episode_id = self.episode_ids[index]
            dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                is_sim = root.attrs['sim']
                original_action_shape = root['/action'].shape
                episode_len = original_action_shape[0]
                if sample_full_episode:
                    start_ts = 0
                else:
                    start_ts = np.random.choice(episode_len)
                #print(f'{start_ts=}')
                # get observation at start_ts only
                compressed = root.attrs.get('compress', False)
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

            if compressed:
                for cam_id, cam_name in enumerate(image_dict.keys()):
                    # un-pad and uncompress
                    padded_compressed_image = image_dict[cam_name]
                    compressed_image = padded_compressed_image
                    image = cv2.imdecode(compressed_image, 1)
                    image_dict[cam_name] = image

            self.is_sim = is_sim
            padded_action = np.zeros(original_action_shape, dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(episode_len)
            is_pad[action_len:] = 1

            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                image = image_dict[cam_name]
                if self.use_augmentation:
                    image = augment_image(image)
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
            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

            return image_data, qpos_data, action_data, is_pad

class TransitionDataset(EpisodicDataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, episode_len, seed=0, use_augmentation=False):
        self.episode_len = episode_len
        super(TransitionDataset).__init__(episode_ids, dataset_dir, camera_names, norm_stats, seed = seed, use_augmentation = use_augmentation)

    def __len__(self):
        return len(self.episode_ids) * self.episode_len

    def __getitem__(self, index):
        episode_index  = index / self.episode_len
        transition_index = index % self.episode_len

        episode_id = self.episode_ids[episode_index]

        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True

        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']

            start_ts = transition_index

            compressed = root.attrs.get('compress', False)
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]

            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

            action = root['/action'][start_ts]

        if compressed:
            for cam_id, cam_name in enumerate(image_dict.keys()):
                # un-pad and uncompress
                padded_compressed_image = image_dict[cam_name]
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_dict[cam_name] = image

        self.is_sim = is_sim

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            image = image_dict[cam_name]
            if self.use_augmentation:
                image = augment_image(image)
            all_cam_images.append(image)
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(action).float()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data

def augment_image(image, threshold = 0.3):
    plt.plot(image)
    raise NotImplementedError



