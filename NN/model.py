from NN.Layers.Patch import PatchEmbedding
from NN.Layers.Encoder import Encoder
from NN.Layers.Decoder import Decoder

import torch
import torch.nn as nn

class VisionTransformerAutoencoder(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_channels=3, embed_dim=512, depth=12,
                 num_heads=8, decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4, activation='gelu', recon_method="method3", dropout=0.1, output_channels=3):
        """
        
        Args:
            image_size (int): "The input image height and width, assumed to be square (default is 256)."
            patch_size (int): "The size of each patch extracted from the image (default is 16)."
            in_channels (int): "Number of input image channels (default is 3 for RGB)."
            embed_dim (int): "Embedding dimension for patch embeddings in the encoder (default is 512)."
            depth (int): "Number of transformer encoder layers (default is 12)."
            num_heads (int): "Number of attention heads for the encoder (default is 8)."
            decoder_embed_dim (int): "Embedding dimension for the decoder (default is 256)."
            decoder_depth (int): "Number of transformer decoder layers (default is 4)."
            decoder_num_heads (int): "Number of attention heads for the decoder (default is 8)."
            mlp_ratio (int): "Multiplier for the feedforward network dimension in transformer layers (default is 4)."
            activation (str): "Activation function to use in the decoder layers (default is 'gelu')."
            recon_method (str): "The selection of reconstruction method (default is 'method3')."
            Dropout (int): "Dropout ratio (0-1) for the drop out layer (default is 0.1)."
        """
        super().__init__()

        assert (image_size % patch_size == 0) and (image_size % patch_size == 0), \
            f"Image dimensions ({image_size}x{image_size}) must be divisible by patch size ({patch_size})"

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        # Encoder
        self.encoder = Encoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            activation=activation,
            dropout=dropout
        )

        # Decoder
        self.decoder = Decoder(
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            image_size=image_size,
            activation=activation,
            patch_size=patch_size,
            recon_method=recon_method,
            dropout=dropout,
            output_channels=output_channels
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): "Input image tensor of shape (B, C, H, W), where B is batch size, C is number of channels, H and W are image dimensions."

        Returns:
            torch.Tensor: "Reconstructed image tensor of shape (B, 3, image_size, image_size)."
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)

        return x
