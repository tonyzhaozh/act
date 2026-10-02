"""Exercise native temporal aggregation with zero-valued action components."""
import importlib
import io
import pickle
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch

# DETR's original modules use imports relative to its source directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detr"))


class RolloutEnv:
    def __init__(self):
        self.task = types.SimpleNamespace(max_reward=1)
        self.rollouts = []

    def timestep(self):
        return types.SimpleNamespace(
            reward=0,
            observation={"qpos": np.zeros(2),
                         "images": {"top": np.zeros((2, 2, 3), dtype=np.uint8)}},
        )

    def reset(self):
        self.rollouts.append([])
        return self.timestep()

    def step(self, action):
        self.rollouts[-1].append(action.copy())
        return self.timestep()


class ChunkPolicy:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def __call__(self, qpos, image):
        chunk = self.chunks[self.calls % len(self.chunks)]
        self.calls += 1
        return torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)

    def load_state_dict(self, state):
        return None

    def cuda(self):
        return self

    def eval(self):
        return self


class TemporalAggregationZeroTest(unittest.TestCase):
    def run_rollouts(self, chunks):
        env = RolloutEnv()
        sim_env = types.ModuleType("sim_env")
        sim_env.BOX_POSE = [None]
        sim_env.make_sim_env = lambda name: env
        old_module = sys.modules.get("sim_env")
        sys.modules["sim_env"] = sim_env
        try:
            episodes = importlib.import_module("imitate_episodes")
            with tempfile.TemporaryDirectory() as folder:
                stats = {name: np.ones(2) if name.endswith("std") else np.zeros(2)
                         for name in ("qpos_mean", "qpos_std", "action_mean", "action_std")}
                with open(Path(folder) / "dataset_stats.pkl", "wb") as file:
                    pickle.dump(stats, file)
                config = {
                    "ckpt_dir": folder, "state_dim": 2, "real_robot": False,
                    "policy_class": "ACT", "onscreen_render": False,
                    "policy_config": {"num_queries": len(chunks[0])},
                    "camera_names": ["top"], "episode_len": len(chunks),
                    "task_name": "test_temporal", "temporal_agg": True,
                }
                with mock.patch.object(episodes, "make_policy", return_value=ChunkPolicy(chunks)), \
                     mock.patch.object(torch, "load", return_value={}), \
                     mock.patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self), \
                     redirect_stdout(io.StringIO()):
                    episodes.eval_bc(config, "test.ckpt", save_episode=False)
        finally:
            if old_module is None:
                sys.modules.pop("sim_env", None)
            else:
                sys.modules["sim_env"] = old_module
        return env.rollouts

    def expected(self, chunks):
        horizon = len(chunks[0])
        actions = []
        for timestep in range(len(chunks)):
            first = max(0, timestep - horizon + 1)
            predictions = np.array([chunks[query][timestep - query]
                                    for query in range(first, timestep + 1)])
            weights = np.exp(-0.01 * np.arange(len(predictions)))
            actions.append(np.sum(predictions * (weights / weights.sum())[:, None], axis=0))
        return np.array(actions)

    def check_rollouts(self, chunks):
        expected = self.expected(chunks)
        rollouts = self.run_rollouts(chunks)
        self.assertEqual(len(rollouts), 50)
        for rollout in rollouts:
            np.testing.assert_allclose(rollout, expected, rtol=1e-6, atol=1e-6)

    def test_zero_components_are_valid_predictions(self):
        self.check_rollouts([
            [[0., 0.], [0., 2.], [3., 0.]],
            [[4., 0.], [0., 5.], [6., 0.]],
            [[6., 0.], [0., 7.], [8., 0.]],
        ])

    def test_expired_queries_are_excluded(self):
        self.check_rollouts([
            [[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]],
            [[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]],
        ])

    def test_nonzero_prediction_weighting_is_preserved(self):
        self.check_rollouts([
            [[1., 2.], [3., 4.], [5., 6.]],
            [[7., 8.], [9., 10.], [11., 12.]],
            [[13., 14.], [15., 16.], [17., 18.]],
        ])


if __name__ == "__main__":
    unittest.main()

