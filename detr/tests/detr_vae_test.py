import pytest
import torch
from torch import nn
from detr.models.position_encoding import PositionEmbeddingSine
from detr.models.backbone import Joiner, ResNetFilmBackbone
from detr.models.transformer import Transformer, IntentTransformer
from detr.models.transformer import TransformerEncoder, TransformerEncoderLayer
from detr.models.detr_vae import DETRVAE, IntentACT

def test_detr_vae():
    # style encoder
    style_encoder_layer = TransformerEncoderLayer(d_model=512,
                                                        nhead=8,
                                                        dim_feedforward=3200,
                                                        dropout=0.1,
                                                        activation='relu',
                                                        normalize_before=False)
    style_encoder_norm = nn.LayerNorm(normalized_shape=512)
    style_encoder = TransformerEncoder(encoder_layer=style_encoder_layer,
                                             num_layers=4,
                                             norm=style_encoder_norm)
    # transformer
    transformer = Transformer(d_model=512,
                              dropout=0.1,
                              nhead=8,
                              dim_feedforward=3200,
                              num_encoder_layers=4,
                              num_decoder_layers=1,
                              normalize_before=False,
                              return_intermediate_dec=True)

    # image encoder backbone
    position_embedding = PositionEmbeddingSine(512//2, normalize=True)
    film_config = {
        'use': False,
        'use_in_layers': [1, 2, 3],
        'task_embedding_dim': 512,
        'film_planes': [64, 128, 256, 512],
    }
    resnet_film_backbone = ResNetFilmBackbone('resnet18', film_config=film_config)
    backbone = Joiner(resnet_film_backbone, position_embedding)
    backbone.num_channels = resnet_film_backbone.num_channels
    backbones = [backbone]

    model = DETRVAE(backbones=backbones,
                    transformer=transformer,
                    encoder=style_encoder,
                    state_dim=14,
                    num_queries=100,
                    camera_names=['top'])

    joint_pos = torch.rand(8, 14)
    image = torch.rand(8, 1, 3, 640, 480)
    a = torch.rand(8, 100, 14)
    is_pad = torch.rand(8, 100).bool()

    a_hat, is_pad_hat, (mu, logvar) = model(qpos=joint_pos,
                                            image=image,
                                            env_state=None,
                                            actions=a,
                                            is_pad=is_pad)

    assert a_hat.shape == torch.Size([8, 100, 14])
    assert is_pad_hat.shape == torch.Size([8, 100, 1])
    assert mu.shape == torch.Size([8, 32])
    assert logvar.shape == torch.Size([8, 32])


def test_intent_act():
    # style encoder
    style_encoder_layer = TransformerEncoderLayer(d_model=512,
                                                nhead=8,
                                                dim_feedforward=3200,
                                                dropout=0.1,
                                                activation='relu',
                                                normalize_before=False)
    style_encoder_norm = nn.LayerNorm(normalized_shape=512)
    style_encoder = TransformerEncoder(encoder_layer=style_encoder_layer,
                                             num_layers=4,
                                             norm=style_encoder_norm)
    # transformer
    transformer = IntentTransformer(d_model=512,
                                    dropout=0.1,
                                    nhead=8,
                                    dim_feedforward=3200,
                                    num_encoder_layers=4,
                                    num_decoder_layers=1,
                                    normalize_before=False,
                                    return_intermediate_dec=True)

    # image encoder backbone
    position_embedding = PositionEmbeddingSine(512//2, normalize=True)
    film_config = {
        'use': False,
        'use_in_layers': [1, 2, 3],
        'task_embedding_dim': 512,
        'film_planes': [64, 128, 256, 512],
    }
    resnet_film_backbone = ResNetFilmBackbone('resnet18', film_config=film_config)
    backbone = Joiner(resnet_film_backbone, position_embedding)
    backbone.num_channels = resnet_film_backbone.num_channels
    backbones = [backbone]

    model = IntentACT(backbones=backbones,
                        intent_transformer=transformer,
                        encoder=style_encoder,
                        state_dim=14,
                        intent_dim=3,
                        num_queries=100,
                        camera_names=['top'])

    joint_pos = torch.rand(8, 14)
    image = torch.rand(8, 1, 3, 640, 480)
    a = torch.rand(8, 100, 14)
    is_pad = torch.rand(8, 100).bool()

    a_hat, a_is_pad_hat, i_hat, i_is_pad_hat, [mu, logvar] = model(qpos=joint_pos,
                                                                    image=image,
                                                                    env_state=None,
                                                                    actions=a,
                                                                    is_pad=is_pad)

    assert a_hat.shape == torch.Size([8, 100, 14])
    assert a_is_pad_hat.shape == torch.Size([8, 100, 1])
    assert i_hat.shape == torch.Size([8, 100, 3])
    assert i_is_pad_hat.shape == torch.Size([8, 100, 1])
    assert mu.shape == torch.Size([8, 32])
    assert logvar.shape == torch.Size([8, 32])

if __name__ == "__main__":
    pytest.main()