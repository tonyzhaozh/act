"""
CVAE model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

import IPython
e = IPython.embed


class CVAE(nn.Module):
    """ This is the CVAE module."""
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        self.backbones = nn.ModuleList(backbones)
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', self.get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = self.reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
            features = features[0] # take the last layer feature
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
        hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)