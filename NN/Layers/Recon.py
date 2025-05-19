"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Zakria Mehmood
# DoC: 2025.04.15
# email: zakriamehmood2001@gmail.com
-----------------------------------------------------------------------------------
# Description: All types of Reconstruction Methods

Note: I've only experimented method 1 and 2. noticible difference among them is spatial quality better in method 2.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.nn.init as init


# linear mapping method
class ReconstructionMethod1(nn.Module):
    def __init__(self, decoder_dim, patch_size, image_size, output_channels):
        super(ReconstructionMethod1, self).__init__()
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.out_channels = output_channels
        # Total patches in the image
        self.num_patches = (image_size // patch_size) ** 2

        self.linear = nn.Linear(decoder_dim, patch_size * patch_size * output_channels, bias=True)

    def forward(self, x):

        batch_size, num_patches, _ = x.shape

        if hasattr(self, 'num_patches'):
            assert num_patches == self.num_patches, f"Expected {self.num_patches} patches but got {num_patches}"
        else:
            self.num_patches = num_patches
            print(
                f"Warning: num_patches was not initialized. Setting to {num_patches} based on input.")

        x = self.linear(x)
        
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      h=int(self.num_patches**0.5), w=int(self.num_patches**0.5),
                      p1=self.patch_size, p2=self.patch_size, c=x.shape[-1] // (self.patch_size * self.patch_size))

        return x


# Convolutional Reconstruction
class ReconstructionMethod2(nn.Module):
    def __init__(self, decoder_dim, patch_size, output_channels):
        super().__init__()
        self.patch_size = patch_size

        self.conv_decoder = nn.Sequential(
            nn.Conv2d(decoder_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, output_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        B, N, C = x.shape
        h = w = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, C, h, w)
        return self.conv_decoder(x)



# Progressive Upsampling Reconstruction
class ReconstructionMethod3(nn.Module):
    def __init__(self, decoder_dim, patch_size, output_channels):
        super().__init__()
        self.patch_size = patch_size

        self.decoder = nn.Sequential(
            nn.Linear(decoder_dim, patch_size * patch_size * 64),
            nn.ReLU(),
            nn.Unflatten(2, (64, patch_size, patch_size)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)



# Residual Reconstruction
class ReconstructionMethod4(nn.Module):
    def __init__(self, decoder_dim, patch_size, output_channels):
        super().__init__()

        class ResBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels)
                )

            def forward(self, x):
                return x + self.conv(x)

        self.decoder = nn.Sequential(
            nn.Linear(decoder_dim, patch_size * patch_size * 64),
            nn.Unflatten(1, (64, patch_size, patch_size)),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, output_channels, 1),
            nn.Tanh()
        )



# Attention-guided Reconstruction
class ReconstructionMethod5(nn.Module):
    def __init__(self, decoder_dim, patch_size, output_channels):
        super().__init__()

        self.query = nn.Linear(decoder_dim, decoder_dim)
        self.key = nn.Linear(decoder_dim, decoder_dim)
        self.value = nn.Linear(decoder_dim, decoder_dim)

        self.final = nn.Sequential(
            nn.Linear(decoder_dim, patch_size * patch_size * output_channels),
            nn.Unflatten(2, (output_channels, patch_size, patch_size))
        )

    def forward(self, x):
        B, N, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention = (q @ k.transpose(-2, -1)) / math.sqrt(C)
        attention = F.softmax(attention, dim=-1)

        x = attention @ v
        return self.final(x)



# Hybrid Reconstruction
class ReconstructionMethod6(nn.Module):
    def __init__(self, decoder_dim, patch_size, output_channels):
        super().__init__()

        self.transformer_part = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=8
        )

        self.cnn_part = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, output_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.transformer_part(x, x)
        B, N, C = x.shape
        h = w = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, h, w)
        return self.cnn_part(x)
