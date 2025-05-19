"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Zakria Mehmood
# DoC: 2025.04.15
# email: zakriamehmood2001@gmail.com
-----------------------------------------------------------------------------------
# Description: Transformer Encoder Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Transformer import CustomTransformerBlock


class Encoder(nn.Module):
    def __init__(self, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, activation='gelu', dropout=0.1):
        """
        Args:
            embed_dim (int): "Dimension of the input embedding for each patch (default is 1024)."
            depth (int): "Number of transformer encoder layers (default is 24)."
            num_heads (int): "Number of attention heads in the multi-head attention mechanism (default is 16)."
            mlp_ratio (int): "Multiplier for the feedforward network dimension in each layer (default is 4)."
            activation (str): "Activation function to use in the decoder layers (default is 'gelu')."
            Dropout (int): "Dropout ratio (0-1) for the drop out layer (default is 0.1)."
        """
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            CustomTransformerBlock(
                embed_dim, num_heads, mlp_ratio, activation, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): "A tensor of shape (B, num_patches, embed_dim) representing the input patch embeddings."
        Returns:
            torch.Tensor: "A normalized tensor of the same shape after processing through the transformer encoder layers."
        """

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

