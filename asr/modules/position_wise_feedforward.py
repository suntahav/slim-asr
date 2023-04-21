# Description: Position-wise Feed-Forward Network
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model = 512, d_ff = 2048, dropout = 0.3) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x)