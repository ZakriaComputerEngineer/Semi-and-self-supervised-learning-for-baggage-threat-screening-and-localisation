
"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Zakria Mehmood
# DoC: 2025.04.15
# email: zakriamehmood2001@gmail.com
-----------------------------------------------------------------------------------
# Description: Patch Embedding Layer by convolutional projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_channels=3, embed_dim=1024):
        """
        Args:
            image_size (int): "The spatial size of the input image (assumed square, default is 256)."
            patch_size (int): "The size of each patch along both height and width dimensions (default is 16)."
            in_channels (int): "Number of input channels in the image (default is 3 for RGB images)."
            embed_dim (int): "The size of the embedding dimension for each patch (default is 1024)."
        """
        super(PatchEmbedding, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        # calculating total number of patches
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Conv projection of the patches
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.pos_embed = nn.Parameter(torch.randn(
            1, self.num_patches, embed_dim))  # Positional Embedding

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): "A 4D input tensor of shape (B, C, H, W), where:
                               B is batch size,
                               C is the number of channels,
                               H and W are the height and width of the image."
        Returns:
            torch.Tensor: "A tensor of shape (B, num_patches, embed_dim) containing the projected patch embeddings with positional embeddings added."
        """
        # Apply conv projection
        x = self.proj(x)
        
        # Reshape using einops: (B, embed_dim, H', W') -> (B, num_patches, embed_dim)
        x = einops.rearrange(x, 'b e h w -> b (h w) e')
        
        # Add positional embeddings
        x = x + self.pos_embed

        return x
