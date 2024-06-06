import pytest
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet101_Weights

from detr.models.resnet_film import BasicBlock, ResNet_FiLM, resnet18, resnet34

# The command to run this script:
# python -m detr.tests.resnet_film_test

def test_ResNet_FiLM_model():
    # resnet film model
    film_config = {
        'use': True,
        'use_in_layers': [1, 2, 3],
        'task_embedding_dim': 512,
        'film_planes': [64, 128, 256, 512],
    }
    model = ResNet_FiLM(BasicBlock, [2, 2, 2, 2], film_config=film_config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/10e6:.2f}M")

    # load pretrained parameters
    weights = ResNet18_Weights.DEFAULT
    model.load_state_dict(weights.get_state_dict(progress=True))

    # FiLM
    film_models = []
    for layer_idx, num_blocks in enumerate(model.layers):
        num_planes = model.film_planes[layer_idx]
        film_model_layer = nn.Linear(
            film_config['task_embedding_dim'], num_blocks * 2 * num_planes)
        film_models.append(film_model_layer)
    film_models = nn.ModuleList(film_models)

    # create mock input
    input = torch.rand(5, 3, 640, 480) # (bs, c, h, w)
    text_emb = torch.rand(5, 512) # (bs, txt_emb_dim)
    film_features = []
    for layer_idx, num_blocks in enumerate(model.layers):
        film_feature = film_models[layer_idx](text_emb)
        film_features.append(film_feature)

    output = model(input, film_features)
    assert output.shape == torch.Size([5, 1000])


def test_resnet18():
    # initialize resnet18
    film_config = {
        'use': True,
        'use_in_layers': [1, 2, 3],
        'task_embedding_dim': 512,
        'film_planes': [64, 128, 256, 512],
    }
    model = resnet18(weights=ResNet18_Weights.DEFAULT, film_config=film_config)

    # FiLM
    film_models = []
    for layer_idx, num_blocks in enumerate(model.layers):
        num_planes = model.film_planes[layer_idx]
        film_model_layer = nn.Linear(
            film_config['task_embedding_dim'], num_blocks * 2 * num_planes)
        film_models.append(film_model_layer)
    film_models = nn.ModuleList(film_models)

    # create mock input
    input = torch.rand(5, 3, 640, 480) # (bs, c, h, w)
    text_emb = torch.rand(5, 512) # (bs, txt_emb_dim)
    film_features = []
    for layer_idx, num_blocks in enumerate(model.layers):
        film_feature = film_models[layer_idx](text_emb)
        film_features.append(film_feature)

    output = model(input, film_features)
    assert output.shape == torch.Size([5, 1000])


def test_resnet34():
    # Initialize resnet34
    film_config = {
        'use': True,
        'use_in_layers': [1, 2, 3],
        'task_embedding_dim': 512,
        'film_planes': [64, 128, 256, 512],
    }
    model = resnet34(weights=ResNet34_Weights.DEFAULT, film_config=film_config)

    # Create film projection model
    film_models = []
    for layer_idx, num_blocks in enumerate(model.layers):
        num_planes = model.film_planes[layer_idx]
        film_model_layer = nn.Linear(
            film_config['task_embedding_dim'], num_blocks * 2 * num_planes)
        film_models.append(film_model_layer)
    film_models = nn.ModuleList(film_models)

    ## mock input
    input = torch.rand(5, 3, 640, 480) # (bs, c, h, w)
    text_emb = torch.rand(5, 512) # (bs, txt_emb_dim)
    film_features = []
    for layer_idx, num_blocks in enumerate(model.layers):
        film_feature = film_models[layer_idx](text_emb)
        film_features.append(film_feature)

    output = model(input, film_features)
    assert output.shape == torch.Size([5, 1000])


if __name__ == "__main__":
    pytest.main()




