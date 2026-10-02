"""Padded action slots do not affect the reconstruction objective."""
import unittest
import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detr"))
from policy import ACTPolicy


class FixedDecoder(nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.predictions = nn.Parameter(predictions.clone())
        self.num_queries = predictions.shape[1]

    def forward(self, qpos, image, env_state, actions, is_pad):
        batch = actions.shape[0]
        mu = actions.new_zeros(batch, 2)
        return self.predictions, is_pad, (mu, torch.zeros_like(mu))


class ACTMaskedReconstructionTest(unittest.TestCase):
    def evaluate(self, predictions, actions, is_pad):
        policy = ACTPolicy.__new__(ACTPolicy)
        nn.Module.__init__(policy)
        policy.model = FixedDecoder(predictions)
        policy.kl_weight = 0.
        batch = actions.shape[0]
        loss = policy(torch.zeros(batch, 2), torch.zeros(batch, 1, 3, 2, 2),
                      actions, is_pad)["l1"]
        return loss, policy.model.predictions

    def test_appending_padding_does_not_dilute_loss(self):
        actions = torch.tensor([[[1., 2.], [3., 4.]]])
        predictions = torch.zeros_like(actions)
        original, _ = self.evaluate(predictions, actions, torch.zeros(1, 2, dtype=torch.bool))
        padded_actions = torch.cat([actions, torch.full((1, 3, 2), 100.)], dim=1)
        padded_predictions = torch.cat([predictions, torch.full((1, 3, 2), -100.)], dim=1)
        padded, _ = self.evaluate(padded_predictions, padded_actions,
                                 torch.tensor([[False, False, True, True, True]]))
        torch.testing.assert_close(original, torch.tensor(2.5))
        torch.testing.assert_close(padded, original)

    def test_valid_coordinate_mean_and_gradient(self):
        actions = torch.tensor([[[1., -2.], [3., 4.]], [[-5., 6.], [100., -100.]]])
        is_pad = torch.tensor([[False, False], [False, True]])
        loss, predictions = self.evaluate(torch.zeros_like(actions), actions, is_pad)
        torch.testing.assert_close(loss, torch.tensor(21. / 6.))
        loss.backward()
        expected = torch.tensor([[[-1., 1.], [-1., -1.]], [[1., -1.], [0., 0.]]]) / 6
        torch.testing.assert_close(predictions.grad, expected)

    def test_fully_padded_batch_has_connected_zero_loss(self):
        actions = torch.full((2, 3, 2), float("nan"))
        loss, predictions = self.evaluate(torch.ones_like(actions), actions,
                                          torch.ones(2, 3, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(predictions.grad, torch.zeros_like(predictions))


if __name__ == "__main__":
    unittest.main()

