import pytest
import torch
from torch import nn
from detr.models.transformer import Transformer, IntentTransformer
from detr.models.transformer import TransformerEncoder, TransformerEncoderLayer
from detr.models.transformer import TextConditionedTransformerEncoder, TextConditionedTransformerEncoderLayer
from detr.models.transformer import TransformerDecoder, TransformerDecoderLayer


# The command to run this script:
# python -m detr.tests.transformer_test

def test_intent_transformer():
    intent_transformer = IntentTransformer(d_model=512,
                                        dropout=0.1,
                                        nhead=8,
                                        dim_feedforward=3200,
                                        num_encoder_layers=4,
                                        num_decoder_layers=1,
                                        normalize_before=False,
                                        return_intermediate_dec=True)

    # create mock input
    image_input = torch.rand(8, 512, 15, 20)
    image_input_mask = None
    latent_input = torch.rand(8, 512)
    proprio_input = torch.rand(8, 512)

    pos_embed = torch.rand(1, 512, 15, 20)
    additional_pos_embed = torch.rand(2, 512)
    query_embed = torch.rand(100, 512)

    # run model
    intent_output, action_output = intent_transformer(src=image_input,
                                                    mask=image_input_mask,
                                                    query_embed=query_embed,
                                                    pos_embed=pos_embed,
                                                    latent_input=latent_input,
                                                    proprio_input=proprio_input,
                                                    additional_pos_embed=additional_pos_embed) # (num_layer, bs, hw, hidden_dim)

    assert intent_output[0].shape == torch.Size([8, 100, 512])
    assert action_output[0].shape == torch.Size([8, 100, 512])


def test_transformer():
    transformer = Transformer(d_model=512,
                              dropout=0.1,
                              nhead=8,
                              dim_feedforward=3200,
                              num_encoder_layers=4,
                              num_decoder_layers=1,
                              normalize_before=False,
                              return_intermediate_dec=True)

    # create mock input
    image_input = torch.rand(8, 512, 15, 20)
    image_input_mask = None
    latent_input = torch.rand(8, 512)
    proprio_input = torch.rand(8, 512)

    pos_embed = torch.rand(1, 512, 15, 20)
    additional_pos_embed = torch.rand(2, 512)
    query_embed = torch.rand(100, 512)

    # run model
    output = transformer(src=image_input,
                         mask=image_input_mask,
                         query_embed=query_embed,
                         pos_embed=pos_embed,
                         latent_input=latent_input,
                         proprio_input=proprio_input,
                         additional_pos_embed=additional_pos_embed) # (num_layer, bs, hw, hidden_dim)

    assert output[0].shape == torch.Size([8, 100, 512])


def test_text_conditioned_transformer_encoder():
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
    assert latent_embed.shape == torch.Size([100, 8, 512])

def test_transformer_encoder():
    # setup model
    transformer_encoder_layer = TransformerEncoderLayer(d_model=512,
                                                        nhead=8,
                                                        dim_feedforward=3200,
                                                        dropout=0.1,
                                                        activation='relu',
                                                        normalize_before=False)
    transformer_encoder_norm = nn.LayerNorm(normalized_shape=512)
    transformer_encoder = TransformerEncoder(encoder_layer=transformer_encoder_layer,
                                             num_layers=4,
                                             norm=transformer_encoder_norm)

    # create mock input
    input_tokens = torch.rand(100, 8, 512)

    # run model
    latent_embed = transformer_encoder(src=input_tokens)
    assert latent_embed.shape == torch.Size([100, 8, 512])


def test_transformer_decoder():
    transformer_decoder_layer = TransformerDecoderLayer(d_model=512,
                                                        nhead=8,
                                                        dim_feedforward=3200,
                                                        dropout=0.1,
                                                        activation='relu',
                                                        normalize_before=False)
    transformer_decoder_norm = nn.LayerNorm(normalized_shape=512)
    transformer_decoder = TransformerDecoder(decoder_layer=transformer_decoder_layer,
                                             num_layers=1,
                                             norm=transformer_decoder_norm,
                                             return_intermediate=True)

    # create mock input
    query = torch.rand(100, 8, 512)
    query_pos = torch.rand(100, 8, 512)

    memory = torch.rand(300, 8, 512)
    memory_pos = torch.rand(300, 8, 512)

    output = transformer_decoder(tgt=query,
                                 query_pos=query_pos,
                                 memory=memory,
                                 pos=memory_pos)

    assert output[0].shape == torch.Size([100, 8, 512])

if __name__ == "__main__":
    pytest.main()