from typing import Optional, Tuple

import torch
from attention import MultiheadAttention
from slimmable import *

__all__ = ["Conformer"]


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


class _ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.
    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
        switches = [0.75,1]
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.switches = switches
        self.layer_norm = SwitchableLayerNorm([int(ele*input_dim) for ele in self.switches])
        
        self.conv1 = SlimmableConv1d(
                [int(ele*input_dim) for ele in self.switches],
                [int(2*num_channels*ele) for ele in self.switches],
                1,
                stride=1,
                padding=0,
                bias=bias,
            )
        self.act1 = torch.nn.GLU(dim=1)
        self.conv2 = SlimmableConv1d(
                [int(ele*num_channels) for ele in self.switches],
                [int(ele*num_channels) for ele in self.switches],
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
                depthwise=True,
            )
        if use_group_norm:
            self.norm = SwitchableGroupnorm(num_groups=1,num_features_list=[int(ele*num_channels) for ele in self.switches])
        else:
            self.norm = SwitchableBatchNorm1d([int(ele*num_channels) for ele in self.switches])
        self.act2 = torch.nn.SiLU()
        self.conv3 = SlimmableConv1d(
                [int(ele*num_channels) for ele in self.switches],
                [int(ele*input_dim) for ele in self.switches],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            )
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, input: torch.Tensor,idx) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.
        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input,idx)
        x = x.transpose(1, 2)
        x = self.conv1(x,idx)
        x = self.act1(x)
        x = self.conv2(x,idx)
        x = self.norm(x,idx)
        x = self.act2(x)
        x = self.conv3(x,idx)
        x = self.drop(x)
        return x.transpose(1, 2)


class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.
    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0,switches=[0.75,1.0]) -> None:
        super().__init__()
        self.switches = switches
        self.ln = SwitchableLayerNorm([int(ele*input_dim) for ele in self.switches])
        self.ff1 = SlimmableLinear([int(ele*input_dim) for ele in self.switches], [int(ele*hidden_dim) for ele in self.switches], bias=True)
        self.act = torch.nn.SiLU()
        self.drop1 = torch.nn.Dropout(dropout)
        self.ff2 = SlimmableLinear([int(ele*hidden_dim) for ele in self.switches], [int(ele*input_dim) for ele in self.switches], bias=True)
        self.drop2 = torch.nn.Dropout(dropout)

    def forward(self, input: torch.Tensor,idx) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.
        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        x = self.ln(input,idx)
        x = self.ff1(x,idx)
        x = self.act(x)
        x = self.drop1(x)
        x = self.ff2(x,idx)
        x = self.drop2(x)
        return x


class ConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.
    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        switches=[0.75,1.0],
    ) -> None:
        super().__init__()
        self.switches = switches
        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = SwitchableLayerNorm([int(ele*input_dim) for ele in self.switches])
        self.self_attn = MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )
        
        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = SwitchableLayerNorm([int(ele*input_dim) for ele in self.switches])
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor,idx) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input,idx)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor],idx) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.
        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input,idx)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x,idx)

        residual = x
        x = self.self_attn_layer_norm(x,idx)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,idx = idx,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x,idx)

        residual = x
        x = self.ffn2(x,idx)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x,idx)
        return x


class Conformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.
    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, lengths: torch.Tensor,idx) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        encoder_padding_mask = _lengths_to_padding_mask(lengths)

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask,idx)
        return x.transpose(0, 1), lengths