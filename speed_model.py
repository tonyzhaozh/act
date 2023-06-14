import torch
from torch import nn
from torch.nn import functional as F

import argparse
import numpy as np
import math

from detr.models.backbone import build_backbone

import IPython
e = IPython.embed


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.03)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src):
        src = self.encoder(src)
        return src

class SpeedDetector(nn.Module):
    def __init__(self, config, backbone='resnet-18', camera_num=1, image_num=3):
        super().__init__()

        assert backbone is not None

        print(config)

        hidden_dim = config['hidden_dim']
        self.backbone = build_backbone(argparse.Namespace(**config))

        self.camera_num = camera_num
        self.image_num = image_num

        # self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.input_proj_robot_state = nn.Linear(14, hidden_dim)

        self.positional_encoding = PositionalEncoding(hidden_dim, 1000)
        self.encoder = TransformerEncoder(hidden_dim, nhead=8, dim_feedforward=2048, num_layers=8)

        self.fc1 = nn.Linear(hidden_dim * image_num * (80 + 1), 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1)

    
    def forward(self, images, robot_state):
        if self.backbone is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            all_qpos_features = []

            batch_size = images.shape[0]
            num_cameras = images.shape[1]
            num_images = images.shape[2]

            assert num_cameras == self.camera_num
            assert num_images == self.image_num

            for cam_id in range(num_cameras):
                for i in range(num_images):
                    features, pos = self.backbone(images[:, cam_id, i])
                    features = features[0]
                    pos = pos[0]

                    features = self.input_proj(features) # (batch_size, hidden_dim, 15, 20)
                    features = features.flatten(start_dim=2) # (batch_size, hidden_dim, 300)
                    #print("image features:", features.shape)

                    all_cam_features.append(features)
                    all_cam_pos.append(pos)
                    
            # proprioception features
            assert robot_state.shape[1] == num_images
            for i in range(num_images):
                qpos = robot_state[:, i]
                #print("qpos:", qpos.shape)
                all_qpos_features.append(self.input_proj_robot_state(qpos).unsqueeze(2))

            src = torch.cat(all_cam_features, axis=2)
            pos = torch.cat(all_cam_pos, axis=2)
            qpos = torch.cat(all_qpos_features, axis=2)

            #print(src.shape)
            #print(qpos.shape)

            all_features = torch.cat([src, qpos], axis=2).permute(0, 2, 1)
            
            #print(all_features.shape)

            x = self.positional_encoding(all_features)
            x = self.encoder(x)
            x = x.flatten(start_dim=1)

            #print(x.shape)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            out = self.fc3(x)

            return out



def build_speed_model_and_optimizer(config):
    policy = SpeedDetector(config)
    policy.cuda()
    optimizer = torch.optim.Adam(
        policy.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )

    return policy, optimizer