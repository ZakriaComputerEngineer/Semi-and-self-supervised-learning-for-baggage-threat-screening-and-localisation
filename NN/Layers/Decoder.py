"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Zakria Mehmood
# DoC: 2025.04.15
# email: zakriamehmood2001@gmail.com
-----------------------------------------------------------------------------------
# Description: Transformer Decoder Layer
"""

import torch
import torch.nn as nn

from .Transformer import CustomTransformerBlock
from .Recon import ReconstructionMethod1, ReconstructionMethod2, ReconstructionMethod3, ReconstructionMethod4, ReconstructionMethod5, ReconstructionMethod6


class Decoder(nn.Module):
    def __init__(self, embed_dim=1024, decoder_embed_dim=512, depth=8, num_heads=16, mlp_ratio=4, image_size=256, activation='gelu', patch_size=16, recon_method="method1", dropout=0.1, output_channels=3):
        """
        Args:
            embed_dim (int): "The embedding dimension of the encoder output (default is 1024)."
            decoder_embed_dim (int): "The embedding dimension for the decoder (default is 512)."
            depth (int): "Number of transformer decoder layers (default is 8)."
            num_heads (int): "Number of attention heads in the multi-head attention mechanism (default is 16)."
            mlp_ratio (int): "Multiplier for the feedforward network dimension in each decoder layer (default is 4)."
            image_size (int): "The size of the reconstructed image (default is 256)."
            activation (str): "Activation function to use in the decoder layers (default is 'gelu')."
            patch_size (int): "The size of each patch along both height and width dimensions (default is 16)."
            recon_method (str): "The selection of reconstruction method (default is 'method1'). 
                                 Use 'method4' for segmentation-optimized reconstruction."
            Dropout (int): "Dropout ratio (0-1) for the drop out layer (default is 0.1)."
        """
        super().__init__()

        # Projection from encoder embedding dimension to decoder embedding dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        # Position embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(torch.zeros(
            1, (image_size // patch_size) ** 2, decoder_embed_dim))

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            CustomTransformerBlock(
                embed_dim=decoder_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                activation=activation,
                dropout=dropout,
                is_decoder=True
            )
            for _ in range(depth)
        ])
        
        # Normalization layer
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Store the reconstruction method name
        self.recon_method = recon_method

        # Select reconstruction method based on the argument
        if recon_method == "method1":
            self.reconstruction = ReconstructionMethod1(
                decoder_dim=decoder_embed_dim,
                patch_size=patch_size,
                image_size=image_size,
                output_channels=output_channels
            )
        elif recon_method == "method2":
            self.reconstruction = ReconstructionMethod2(
                decoder_dim=decoder_embed_dim,
                patch_size=patch_size,
                output_channels=output_channels
            )
        elif recon_method == "method3":
            self.reconstruction = ReconstructionMethod3(
                decoder_dim=decoder_embed_dim,
                patch_size=patch_size,
                output_channels=output_channels
            )
        elif recon_method == "method4":
            # Segmentation-optimized reconstruction method
            self.reconstruction = ReconstructionMethod4(
                decoder_dim=decoder_embed_dim,
                patch_size=patch_size,
                output_channels=output_channels
            )
        elif recon_method == "method5":
            self.reconstruction = ReconstructionMethod5(
                decoder_dim=decoder_embed_dim,
                patch_size=patch_size,
                output_channels=output_channels
            )
        elif recon_method == "method6":
            self.reconstruction = ReconstructionMethod6(
                decoder_dim=decoder_embed_dim,
                patch_size=patch_size,
                output_channels=output_channels
            )

        else:
            raise ValueError(
                f"Unknown reconstruction method: {recon_method}. Choose from 'method1', 'method2', 'method3', 'method4', 'method5' or 'method6'.")


    def forward(self, x, memory=None):
        # Project from encoder dimension to decoder dimension
        x = self.decoder_embed(x)

        # Add position embeddings
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x, memory) if memory is not None else block(x, x)

        # Apply normalization
        x = self.decoder_norm(x)

        # Apply reconstruction method to get the final output
        output = self.reconstruction(x)

        return output
