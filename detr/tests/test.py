import torch
from torch import nn
from detr.models.transformer import Transformer, IntentTransformer
from detr.models.transformer import TextConditionedTransformerEncoderLayer, TextConditionedTransformerEncoder
from detr.models.transformer import TransformerDecoder, TransformerDecoderLayer


# The command to run this script:
# python -m detr.tests.transformer_test


# setup model
transformer_encoder_layer = TextConditionedTransformerEncoderLayer(d_model=512,
                                                                    nhead=8,
                                                                    dim_feedforward=3200,
                                                                    dropout=0.1,
                                                                    activation='relu',
                                                                    normalize_before=False)
transformer_encoder_norm = nn.LayerNorm(normalized_shape=512)
transformer_encoder = TextConditionedTransformerEncoder(encoder_layer=transformer_encoder_layer,
                                                        num_layers=4,
                                                        norm=transformer_encoder_norm)

# create mock input
input_tokens = torch.rand(100, 8, 512)
text_condition = torch.rand(8, 1, 512)
pos_embed = torch.rand(100, 1, 512)

# run model
latent_embed = transformer_encoder_layer(src=input_tokens, text_condition=text_condition, pos=pos_embed)
print(latent_embed.shape)
