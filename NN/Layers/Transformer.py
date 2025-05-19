"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Zakria Mehmood
# DoC: 2025.04.15
# email: zakriamehmood2001@gmail.com
-----------------------------------------------------------------------------------
# Description: Custom Coded Transformer Block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utilities.training_utils import get_activation_function

class CustomTransformerBlock(nn.Module):
    """
    Custom transformer encoder block with multi-head self-attention and a feed-forward network.

    Args:
        embed_dim (int): The embedding dimension of the input.
        num_heads (int): The number of attention heads.
        mlp_ratio (float): The ratio of the size of the feed-forward network to the embedding size.
        activation (str): The activation function to use in the feed-forward network.
        is_decoder(bool): Whether this block is part of a decoder.

    Methods:
        forward(src, src_mask=None, src_key_padding_mask=None):
            Args:
                src (torch.Tensor): The input tensor.
                src_mask (torch.Tensor, optional): The mask for the input tensor.
                src_key_padding_mask (torch.Tensor, optional): The key padding mask for the input tensor.

            Returns:
                torch.Tensor: The output tensor after self-attention and feed-forward pass.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio, activation, dropout, is_decoder=False):
        super(CustomTransformerBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.is_decoder = is_decoder
        
        if is_decoder:
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
            self.norm3 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            get_activation_function(activation),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)            
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, memory=None, src_mask=None, memory_mask=None,src_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Forward pass for the encoder block.

        Args:
            src (torch.Tensor): The input tensor.
            src_mask (torch.Tensor, optional): The mask for the input tensor.
            src_key_padding_mask (torch.Tensor, optional): The key padding mask for the input tensor.

        Returns:
            torch.Tensor: The output tensor after self-attention and feed-forward pass.
        """
        
        src = self.norm1(src)  # Layer normalization
        
        attn_output, _ = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        
        src = src + self.dropout(attn_output)  # Add residual connection
        
        src = self.norm2(src)  # Layer normalization
        
        if self.is_decoder:
            cross_output, _ = self.cross_attn(src, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
            src = src + self.dropout(cross_output)
            src = self.norm3(src)  # Layer normalization
            
        ffn_output = self.ffn(src)
        
        return ffn_output

