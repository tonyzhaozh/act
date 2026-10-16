"""Recorded timesteps, not artificial padding, define demonstration statistics."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from torch.utils.data import DataLoader

from utils import EpisodicDataset, get_norm_stats


class VariableEpisodeTest(unittest.TestCase):

    def write_episodes(self, folder, lengths, is_sim=True):
        recorded = []
        for episode, length in enumerate(lengths):
            qpos = (np.arange(length * 3).reshape(length, 3) * .2 + episode * 4).astype(np.float32)
            action = (qpos * .5 + np.array([1., -2., 3.])).astype(np.float32)
            with h5py.File(Path(folder) / f'episode_{episode}.hdf5', 'w') as root:
                root.attrs['sim'] = is_sim
                root.create_dataset('action', data=action)
                root.create_dataset('observations/qpos', data=qpos)
                root.create_dataset('observations/qvel', data=np.zeros_like(qpos))
                root.create_dataset('observations/images/top', data=np.full((length, 2, 3, 3), 120, np.uint8))
            recorded.append((qpos, action))
        return recorded

    def assert_stats(self, stats, episodes):
        for key, index in (('qpos', 0), ('action', 1)):
            # Independent aggregate over actual recorded frames, with the
            # existing Bessel-corrected estimator and standard-deviation floor.
            samples = np.concatenate([episode[index] for episode in episodes])
            np.testing.assert_allclose(stats[f'{key}_mean'], np.mean(samples, axis=0), rtol=2e-6, atol=1e-6)
            expected_std = np.maximum(np.std(samples, axis=0, ddof=1), .01)
            np.testing.assert_allclose(stats[f'{key}_std'], expected_std, rtol=2e-6, atol=1e-6)

    def test_statistics_weight_recorded_timesteps_of_unequal_episodes(self):
        with tempfile.TemporaryDirectory() as folder:
            episodes = self.write_episodes(folder, [2, 5, 3])
            self.assert_stats(get_norm_stats(folder, 3), episodes)

    def test_real_dataloader_collates_unequal_episodes_and_preserves_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            episodes = self.write_episodes(folder, [2, 5])
            stats = get_norm_stats(folder, 2)
            with mock.patch('utils.np.random.choice', return_value=0):
                dataset = EpisodicDataset([0, 1], folder, ['top'], stats)
                image, qpos, actions, padding = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
            self.assertEqual(tuple(actions.shape), (2, 5, 3))
            self.assertEqual(tuple(image.shape), (2, 1, 3, 2, 3))
            np.testing.assert_array_equal(padding.numpy(), [[False, False, True, True, True], [False]*5])
            for index, (positions, recorded_actions) in enumerate(episodes):
                expected = (recorded_actions - stats['action_mean']) / stats['action_std']
                np.testing.assert_allclose(actions[index, :len(recorded_actions)].numpy(), expected, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(qpos[index].numpy(), (positions[0] - stats['qpos_mean']) / stats['qpos_std'], rtol=1e-6, atol=1e-6)
            self.assert_stats(stats, episodes)

    def test_tail_sampling_keeps_actual_action_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            self.write_episodes(folder, [2, 5])
            stats = get_norm_stats(folder, 2)
            for is_sim in (False, True):
                for episode in range(2):
                    with h5py.File(Path(folder) / f'episode_{episode}.hdf5', 'r+') as root:
                        root.attrs['sim'] = is_sim
                with mock.patch('utils.np.random.choice', return_value=1):
                    dataset = EpisodicDataset([0, 1], folder, ['top'], stats)
                    masks = [dataset[index][3].numpy() for index in range(2)]
                expected_counts = [1, 4] if is_sim else [2, 5]
                for mask, count in zip(masks, expected_counts):
                    np.testing.assert_array_equal(mask, np.arange(5) >= count)

    def test_equal_length_demonstrations_keep_statistics_and_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            episodes = self.write_episodes(folder, [4, 4])
            stats = get_norm_stats(folder, 2)
            self.assert_stats(stats, episodes)
            with mock.patch('utils.np.random.choice', return_value=0):
                dataset = EpisodicDataset([0, 1], folder, ['top'], stats)
                _, _, actions, padding = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
            self.assertEqual(tuple(actions.shape), (2, 4, 3))
            self.assertFalse(padding.any().item())

    def test_single_recorded_timestep_has_finite_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            episodes = self.write_episodes(folder, [1])
            stats = get_norm_stats(folder, 1)
            for key, index in (('qpos', 0), ('action', 1)):
                np.testing.assert_array_equal(stats[f'{key}_mean'], episodes[0][index][0])
                np.testing.assert_allclose(stats[f'{key}_std'], np.full(3, .01))
            dataset = EpisodicDataset([0], folder, ['top'], stats)
            _, qpos, actions, padding = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
            np.testing.assert_array_equal(qpos.numpy(), np.zeros((1, 3)))
            np.testing.assert_array_equal(actions.numpy(), np.zeros((1, 1, 3)))
            self.assertFalse(padding.any().item())


if __name__ == '__main__':
    unittest.main()
