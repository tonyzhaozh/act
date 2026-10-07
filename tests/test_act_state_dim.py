"""Real ACT image/latent transformers accept the declared robot state width."""
import sys
from pathlib import Path
import unittest

import torch
import torchvision

# Original DETR imports use the source directory as a package root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'detr'))
from detr.models.backbone import BackboneBase, FrozenBatchNorm2d, Joiner
from detr.models.detr_vae import DETRVAE
from detr.models.position_encoding import PositionEmbeddingSine
from detr.models.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer


class ACTStateDimTest(unittest.TestCase):
    def model(self, state_dim):
        torch.manual_seed(29)
        # Real randomly initialized ResNet; no pretrained weights or downloads.
        resnet = torchvision.models.resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
        backbone = BackboneBase(resnet, True, 512, False)
        joiner = Joiner(backbone, PositionEmbeddingSine(8, normalize=True))
        joiner.num_channels = backbone.num_channels
        transformer = Transformer(d_model=16, nhead=2, num_encoder_layers=1,
                                  num_decoder_layers=1, dim_feedforward=24, dropout=0.)
        encoder = TransformerEncoder(TransformerEncoderLayer(16, 2, 24, dropout=0.), 1)
        return DETRVAE([joiner], transformer, encoder, state_dim=state_dim,
                       num_queries=3, camera_names=['top'])

    def test_training_forward_and_latent_gradient_at_nondefault_state_width(self):
        for state_dim in (7, 21):
            with self.subTest(state_dim=state_dim):
                model = self.model(state_dim)
                qpos = torch.randn(2, state_dim, requires_grad=True)
                actions = torch.randn(2, 3, state_dim, requires_grad=True)
                output, padding, (mu, logvar) = model(qpos, torch.randn(2, 1, 3, 32, 32), None,
                                                     actions, torch.zeros(2, 3, dtype=torch.bool))
                self.assertEqual(tuple(output.shape), (2, 3, state_dim))
                self.assertEqual(tuple(padding.shape), (2, 3, 1))
                self.assertEqual(tuple(mu.shape), (2, 32))
                self.assertEqual(tuple(logvar.shape), (2, 32))
                (output.square().mean() + mu.square().mean() + logvar.square().mean()).backward()
                for value in (qpos, actions):
                    self.assertTrue(torch.isfinite(value.grad).all().item())
                    self.assertGreater(value.grad.abs().sum().item(), 0.)
                for layer in (model.encoder_joint_proj, model.encoder_action_proj, model.input_proj_robot_state):
                    self.assertEqual(layer.in_features, state_dim)
                    self.assertTrue(torch.isfinite(layer.weight.grad).all().item())

    def test_prior_inference_accepts_nondefault_proprioceptive_width(self):
        model = self.model(7).eval()
        with torch.no_grad():
            output, padding, (mu, logvar) = model(torch.zeros(2, 7), torch.zeros(2, 1, 3, 32, 32), None)
        self.assertEqual(tuple(output.shape), (2, 3, 7))
        self.assertEqual(tuple(padding.shape), (2, 3, 1))
        self.assertIsNone(mu)
        self.assertIsNone(logvar)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_default_width_preserves_checkpoint_shapes_and_full_forward(self):
        first = self.model(14)
        second = self.model(14)
        second.load_state_dict(first.state_dict(), strict=True)
        for layer in (first.encoder_joint_proj, first.encoder_action_proj, first.input_proj_robot_state):
            self.assertEqual(tuple(layer.weight.shape), (16, 14))
        qpos = torch.randn(2, 14)
        image = torch.randn(2, 1, 3, 32, 32)
        actions = torch.randn(2, 3, 14)
        mask = torch.zeros(2, 3, dtype=torch.bool)
        torch.manual_seed(41)
        first_output = first(qpos, image, None, actions, mask)
        torch.manual_seed(41)
        second_output = second(qpos, image, None, actions, mask)
        for index in (0, 1):
            torch.testing.assert_close(first_output[index], second_output[index], rtol=0, atol=0)
        for first_latent, second_latent in zip(first_output[2], second_output[2]):
            torch.testing.assert_close(first_latent, second_latent, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
