import argparse

import pytest
import torch
from detr.models.backbone import ResNetFilmBackbone, build_film_backbone

# The command to run this script:
# python -m detr.tests.backbone_test
def test_ResNet_FiLM_Backbone():
    # setup config
    film_config = {
        'use': True,
        'use_in_layers': [1, 2, 3],
        'task_embedding_dim': 512,
        'film_planes': [64, 128, 256, 512],
    }

    # setup model
    resnet_film_backbone = ResNetFilmBackbone('resnet18', film_config=film_config)

    # setup mock model input
    input = torch.rand(5, 3, 640, 480)  # (bs, c, h, w)
    text_emb = torch.rand(5, 512)  # (bs, txt_emb_dim)

    # run model
    output = resnet_film_backbone(input, task_emb = text_emb)

    # check
    assert output.shape == torch.Size([5, 512, 20, 15])

def test_build_film_backbone():
    def dict_to_namespace(d):
        namespace = argparse.Namespace()
        for key, value in d.items():
            setattr(namespace, key, value)
        return namespace

    config = {
        'hidden_dim': 512,
        'position_embedding': 'sine',
        'backbone': 'resnet18',
    }
    args = dict_to_namespace(config)
    resnet_film_backbone = build_film_backbone(args)

    # setup mock model input
    input = torch.rand(5, 3, 640, 480)  # (bs, c, h, w)
    text_emb = torch.rand(5, 512)  # (bs, txt_emb_dim)

    # run model
    output, pos = resnet_film_backbone(input, task_emb = text_emb)
    assert output[0].shape == torch.Size([5, 512, 20, 15])
    assert pos[0].shape == torch.Size([1, 512, 20, 15])

if __name__ == "__main__":
    pytest.main()