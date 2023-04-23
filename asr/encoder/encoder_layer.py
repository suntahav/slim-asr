import torch
import torch.nn as nn
import sys
sys.path.append("..")
from modules.attention import MultiHeadAttention
from modules.position_wise_feedforward import PositionWiseFeedForward

class TransformerTransducerEncoderLayer(nn.Module):
    def __init__(self, d_model = 512, nhead = 8, dim_feedforward = 2048, dropout = 0.1, activation = "relu") -> None:
        super(TransformerTransducerEncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.encoder_dropout = nn.Dropout(dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, dim_feedforward, dropout)
        
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = self.layer_norm(inputs)
        self_attn_output, self_attn_output_weights = self.self_attn(inputs, inputs, inputs, mask)
        self_attn_output += inputs
        
        self_attn_output = self.layer_norm(self_attn_output)
        ff_output = self.feed_forward(self_attn_output)
        output = self.encoder_dropout(ff_output + self_attn_output)
        
        return output, self_attn_output_weights
    
        
        