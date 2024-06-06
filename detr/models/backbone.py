# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Any, Dict, List, Mapping, Optional

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .resnet_film import resnet18 as resnet18_film
from .resnet_film import resnet34 as resnet34_film

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


# ====  ResNet Backbone ====
class ResNetFilmBackbone(nn.Module):
    def __init__(self, embedding_name: str, pretrained: bool = False,
                 film_config: Optional[Mapping[str, Any]] = None):
        super().__init__()
        self._pretrained = pretrained
        weights = 'IMAGENET1K_V1' if pretrained else None
        if embedding_name in ('resnet34_film', 'resnet34'):
            backbone = resnet34_film(weights=weights, film_config=film_config, pretrained=pretrained)
            embedding_dim = 512
        elif embedding_name in ('resnet18_film', 'resnet18'):
            backbone = resnet18_film(weights=weights, film_config=film_config, pretrained=pretrained)
            embedding_dim = 512
        else:
            raise NotImplementedError

        # remove the last two layers of resnet
        self.resnet_film_model = backbone
        self.resnet_film_model.fc = nn.Identity()
        self.resnet_film_model.avgpool = nn.Identity()

        self._embedding_dim = embedding_dim

        self.num_channels = self._embedding_dim

        # FiLM config
        self.film_config = film_config
        if film_config is not None and film_config['use']:
            film_models = []
            for layer_idx, num_blocks in enumerate(self.resnet_film_model.layers):
                if layer_idx in film_config['use_in_layers']:
                    num_planes = self.resnet_film_model.film_planes[layer_idx]
                    film_model_layer = nn.Linear(
                        film_config['task_embedding_dim'], num_blocks * 2 * num_planes)
                else:
                    film_model_layer = None
                film_models.append(film_model_layer)

            self.film_models = nn.ModuleList(film_models)

    def forward(self, x, texts: Optional[List[str]] = None, task_emb: Optional[torch.Tensor] = None, **kwargs):
        film_outputs = None
        if self.film_config is not None and self.film_config['use']:
            film_outputs = []
            for layer_idx, num_blocks in enumerate(self.resnet_film_model.layers):
                if self.film_config['use'] and self.film_models[layer_idx] is not None:
                    film_features = self.film_models[layer_idx](task_emb)
                else:
                    film_features = None
                film_outputs.append(film_features)
        return self.resnet_film_model(x, film_features=film_outputs, flatten=False)

    @property
    def embed_dim(self):
        return self._embedding_dim


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, task_emb: Optional[Any] = None):
        if task_emb is not None:
            xs = self[0](tensor_list, task_emb=task_emb)
            # Make a dictionary out of the last layer outputs since we don't have IntermediateLayerGetter
            xs = {'0': xs}
        else:
            xs = self[0](tensor_list)
            xs = {'0': xs}
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_film_backbone(args):
    position_embedding = build_position_encoding(args)
    film_config = {
        'use': True,
        'use_in_layers': [1, 2, 3],
        'task_embedding_dim': 512,
        'film_planes': [64, 128, 256, 512],
    }
    backbone = ResNetFilmBackbone(args.backbone, film_config=film_config)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model