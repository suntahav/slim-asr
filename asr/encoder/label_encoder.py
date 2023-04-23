import torch
from torch import Tensor
import torch.nn as nn
import sys
sys.path.append("..")
from modules.positional_encoding import PositionalEncoding
from modules.mask import get_attn_pad_mask, get_attn_subsequent_mask
from .encoder_layer import TransformerTransducerEncoderLayer
from typing import Tuple
import numpy as np

class TransformerTransducerLabelEncoder(nn.Module):
    def __init__(
        self,
        device :torch.device,
        num_classes: int,
        model_dim: int = 512,
        d_ff: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_positional_length: int = 5000,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
    ) -> None:
        super(TransformerTransducerLabelEncoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(num_classes, model_dim)
        self.scale = np.sqrt(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_positional_length)
        self.input_dropout = nn.Dropout(p=dropout)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.encoder_layers = nn.ModuleList(
            [TransformerTransducerEncoderLayer(model_dim, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )
        
    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = inputs.size(0)

        if len(inputs.size()) == 1:  # validate, evaluation
            inputs = inputs.unsqueeze(1)
            target_lengths = inputs.size(1)

            outputs = self.forward_step(
                encoder_inputs=inputs,
                encoder_input_lengths=input_lengths,
                positional_encoding_length=target_lengths,
            )

        else:  # train
            target_lengths = inputs.size(1)

            outputs = self.forward_step(
                encoder_inputs=inputs,
                encoder_input_lengths=input_lengths,
                positional_encoding_length=target_lengths,
            )

        return outputs, input_lengths
    
    def forward_step(
        self,
        encoder_inputs: torch.Tensor,
        encoder_input_lengths: torch.Tensor,
        positional_encoding_length: int = 1,
    ) -> torch.Tensor:
        enc_self_attn_pad_mask = get_attn_pad_mask(encoder_inputs, encoder_input_lengths, encoder_inputs.size(1))
        enc_self_attn_subsequent_mask = get_attn_subsequent_mask(encoder_inputs)
        self_attn_mask = torch.gt((enc_self_attn_pad_mask + enc_self_attn_subsequent_mask), 0)

        embedding_output = self.embedding(encoder_inputs) * self.scale
        positional_encoding_output = self.positional_encoding(positional_encoding_length)
        inputs = embedding_output + positional_encoding_output

        outputs = self.input_dropout(inputs)

        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)

        return outputs