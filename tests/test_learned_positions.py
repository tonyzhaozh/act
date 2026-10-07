import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'detr'))

from types import SimpleNamespace

import torchvision

from detr.models.backbone import BackboneBase, FrozenBatchNorm2d, Joiner
from detr.models.detr_vae import DETRVAE
from detr.models.position_encoding import PositionEmbeddingLearned, PositionEmbeddingSine, build_position_encoding
from detr.models.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer
from util.misc import NestedTensor


class LearnedPositionsTest(unittest.TestCase):
    def model(self):
        torch.manual_seed(71)
        # Genuine ResNet and original ACT transformer; no downloads or substitutes.
        resnet = torchvision.models.resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
        backbone = BackboneBase(resnet, True, 512, False)
        positions = build_position_encoding(SimpleNamespace(hidden_dim=16, position_embedding='learned'))
        joiner = Joiner(backbone, positions)
        joiner.num_channels = backbone.num_channels
        transformer = Transformer(d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=24, dropout=0.)
        encoder = TransformerEncoder(TransformerEncoderLayer(16, 2, 24, dropout=0.), 1)
        return DETRVAE([joiner], transformer, encoder, state_dim=14, num_queries=3, camera_names=['top'])

    def test_raw_backbone_features_and_nested_inputs_have_identical_absolute_positions(self):
        module = PositionEmbeddingLearned(8)
        x = torch.randn(3, 5, 3, 4)
        actual = module(x)
        expected = torch.empty(3, 16, 3, 4)
        for row in range(3):
            for col in range(4):
                expected[:, :, row, col] = torch.cat([module.col_embed.weight[col], module.row_embed.weight[row]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(module(NestedTensor(x, None)), expected, rtol=0, atol=0)

    def test_real_training_configuration_handles_multiple_batch_elements(self):
        model = self.model()
        qpos = torch.randn(2, 14, requires_grad=True)
        actions = torch.randn(2, 3, 14, requires_grad=True)
        output, padding, (mu, logvar) = model(qpos, torch.randn(2, 1, 3, 32, 32), None, actions, torch.zeros(2, 3, dtype=torch.bool))
        self.assertEqual(tuple(output.shape), (2, 3, 14))
        self.assertEqual(tuple(padding.shape), (2, 3, 1))
        (output.square().mean() + mu.square().mean() + logvar.square().mean()).backward()
        for parameter in model.backbones[0][1].parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(qpos.grad).all().item())
        self.assertTrue(torch.isfinite(actions.grad).all().item())

    def test_prior_inference_is_invariant_to_repeating_the_batch(self):
        model = self.model().eval()
        qpos = torch.randn(1, 14)
        image = torch.randn(1, 1, 3, 32, 32)
        with torch.no_grad():
            one = model(qpos, image, None)[0]
            repeated = model(qpos.repeat(3, 1), image.repeat(3, 1, 1, 1, 1), None)[0]
        torch.testing.assert_close(repeated, one.repeat(3, 1, 1), rtol=2e-5, atol=2e-6)

    def test_sine_singleton_and_expanded_positions_are_equivalent_transformer_inputs(self):
        transformer = Transformer(d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=24, dropout=0.).eval()
        source = torch.randn(3, 16, 2, 3)
        singleton = PositionEmbeddingSine(8, normalize=True)(source)
        query, latent, proprio, extra = torch.randn(4, 16), torch.randn(3, 16), torch.randn(3, 16), torch.randn(2, 16)
        one = transformer(source, None, query, singleton, latent, proprio, extra)
        expanded = transformer(source, None, query, singleton.expand(3, -1, -1, -1), latent, proprio, extra)
        torch.testing.assert_close(one, expanded, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
