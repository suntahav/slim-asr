from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DotProductAttention(nn.Module):
    
    def __init__(self, key_dim: int) -> None:
        super().__init__()
        self.sqrt_key_dim = np.sqrt(key_dim)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        # query: (batch_size, query_len, key_dim)
        # key: (batch_size, key_len, key_dim)
        # value: (batch_size, key_len, value_dim)
        # return: (batch_size, query_len, value_dim), (batch_size, query_len, key_len)
        # (batch_size, query_len, key_len)
        key = key.transpose(1, 2)
        attention_weights = torch.bmm(query, key) / self.sqrt_key_dim
        # (batch_size, query_len, key_len)
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask, -np.inf)
        # (batch_size, query_len, key_len)
        attention_weights = F.softmax(attention_weights, dim=-1)
        # (batch_size, query_len, value_dim)
        attention_output = torch.bmm(attention_weights, value)
        return attention_output, attention_weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads 
        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        # self.output_projection = nn.Linear(model_dim, model_dim)
        self.attention = DotProductAttention(self.head_dim)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        batch_size = query.size(0)
        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.head_dim)
        key = self.key_projection(key).view(batch_size, -1, self.num_heads, self.head_dim)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        query = query.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.head_dim)
        key = key.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.head_dim)
        value = value.permute(0, 2, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.head_dim)
        
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
            
        context, attention_weights = self.attention(query, key, value, mask)
        
        context = context.view(batch_size, self.num_heads, -1, self.head_dim)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.model_dim)
        
        return context, attention_weights