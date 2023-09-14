import argparse
import json
import torch
import os

from act.models.cvae import CVAE
from act.models.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer
from act.models.backbone import Backbone, Joiner
from act.models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned


class ACTBuilder:
    @classmethod
    def get_args_parser(cls, args_list):
        # Load the config file
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(curr_dir, "config", "initial.json")
        with open(file_path, "r") as file:
            config = json.load(file)

        parser = argparse.ArgumentParser('Act parameters', add_help=False)

        # Automatically add arguments from the config
        for key, value in config.items():
            if isinstance(value, bool):
                parser.add_argument(f'--{key}', action='store_true' if not value else 'store_false')
            elif value is None:
                parser.add_argument(f'--{key}', required=True)
            else:
                parser.add_argument(f'--{key}', default=value, type=type(value))
        # Modify this line:
        return parser, args_list

    @classmethod
    def build(cls, args_override):
        parser, args_list = cls.get_args_parser([])
        args = parser.parse_args(args_list)

        for k, v in args_override.items():
            setattr(args, k, v)

        # From image
        backbones = []
        backbone = cls.build_backbone(args)
        backbones.append(backbone)

        transformer = cls.build_transformer(args)

        encoder = cls.build_encoder(args)

        model = CVAE(
            backbones,
            transformer,
            encoder,
            state_dim=args.state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
        )
        model.cuda()

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters/1e6,))

        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

        return model, optimizer

    @classmethod
    def build_backbone(cls, args):
        position_embedding = cls.build_position_encoding(args)
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_channels
        return model

    @classmethod
    def build_position_encoding(cls, args):
        N_steps = args.hidden_dim // 2
        if args.position_embedding in ('v2', 'sine'):
            # TODO find a better way of exposing other arguments
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        elif args.position_embedding in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned(N_steps)
        else:
            raise ValueError(f"not supported {args.position_embedding}")

        return position_embedding

    @staticmethod
    def build_encoder(args):
        d_model = args.hidden_dim  # 256
        dropout = args.dropout  # 0.1
        nhead = args.nheads  # 8
        dim_feedforward = args.dim_feedforward  # 2048
        num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
        normalize_before = args.pre_norm  # False
        activation = "relu"

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = torch.nn.LayerNorm(d_model) if normalize_before else None
        encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        return encoder

    @staticmethod
    def build_transformer(args):
        return Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
