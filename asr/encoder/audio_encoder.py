import torch
from torch import nn
import sys
sys.path.append("..")
from modules.positional_encoding import PositionalEncoding
from modules.mask import get_attn_pad_mask
from .encoder_layer import TransformerTransducerEncoderLayer

from typing import Tuple

class TransformerTransducerAudioEncoder(nn.Module):
    # Description: Transformer Encoder for Transformer Transducer
    def __init__(
        self,
        input_size: int = 80,
        model_dim: int = 512,
        d_ff: int = 2048,
        num_layers: int = 18,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_positional_length: int = 5000,
    ) -> None:
        super(TransformerTransducerAudioEncoder, self).__init__()
        self.input_size = input_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_positional_length)
        self.input_fc = nn.Linear(input_size, model_dim)
        self.encoder_layers = nn.ModuleList(
            [TransformerTransducerEncoderLayer(model_dim, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward propagate a `inputs` for audio encoder.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
            ** input_lengths**(Tensor):  ``(batch)``
        """
        seq_len = inputs.size(1)

        self_attn_mask = get_attn_pad_mask(inputs, input_lengths, seq_len)

        inputs = self.input_fc(inputs) + self.positional_encoding(seq_len)
        outputs = self.input_dropout(inputs)

        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)

        return outputs, input_lengths