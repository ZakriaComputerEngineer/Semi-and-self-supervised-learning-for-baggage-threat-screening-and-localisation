"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Zakria Mehmood
# DoC: 2025.04.15
# email: zakriamehmood2001@gmail.com
-----------------------------------------------------------------------------------
# Description: Different types of loss functions for image reconstruction and segmentation task.

Note: only combinational loss & perceptual loss are used in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Custom loss functions for image reconstruction
class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    """

    def __init__(self, layers=[4, 9], weights=[1.0, 1.0]):
        """
        Args:
            layers (list): Indices of VGG layers to use for feature extraction
            weights (list): Weights for each layer's contribution to the loss
        """
        super().__init__()
        # Load pretrained VGG16 model
        vgg = models.vgg16(pretrained=True).features

        # Create slices for feature extraction
        self.slices = nn.ModuleList()
        start_idx = 0
        for end_idx in layers:
            self.slices.append(nn.Sequential(
                *list(vgg.children())[start_idx:end_idx]))
            start_idx = end_idx

        self.weights = weights

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        # Register mean and std for normalization
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, img):
        """Normalize image for VGG"""
        return (img - self.mean) / self.std

    def forward(self, input, target):
        """
        Args:
            input (torch.Tensor): Predicted image
            target (torch.Tensor): Target image

        Returns:
            torch.Tensor: Perceptual loss
        """
        # Make sure input and target are on the same device as the model
        device = next(self.parameters()).device
        input = input.to(device)
        target = target.to(device)

        # Normalize inputs
        input = self._normalize(input)
        target = self._normalize(target)

        # Extract features and compute loss
        loss = 0.0
        input_features = input
        target_features = target

        for i, slice in enumerate(self.slices):
            input_features = slice(input_features)
            target_features = slice(target_features)
            loss += self.weights[i] * \
                F.mse_loss(input_features, target_features)

        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    """

    def __init__(self, window_size=11, size_average=True):
        """
        Args:
            window_size (int): Size of the Gaussian window
            size_average (bool): Whether to average the loss over spatial dimensions
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size)

    def _create_window(self, window_size):
        """Create a Gaussian window"""
        _1D_window = torch.Tensor([1.0]).expand(window_size).unsqueeze(1)
        _1D_window = _1D_window * _1D_window.t()
        _1D_window = _1D_window / _1D_window.sum()
        window = _1D_window.unsqueeze(0).unsqueeze(0)
        return window

    def forward(self, img1, img2):
        """
        Args:
            img1 (torch.Tensor): First image
            img2 (torch.Tensor): Second image

        Returns:
            torch.Tensor: 1 - SSIM (as a loss, lower is better)
        """
        # Move window to same device as input
        window = self.window.to(img1.device)
        window = window.expand(
            img1.size(1), 1, self.window_size, self.window_size)

        # Calculate means
        mu1 = F.conv2d(img1, window, padding=self.window_size //
                       2, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=self.window_size //
                       2, groups=img2.size(1))

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Calculate variances and covariance
        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.window_size//2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.window_size//2, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.window_size//2, groups=img1.size(1)) - mu1_mu2

        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
            ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # Return 1 - SSIM to convert to a loss (lower is better)
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class CombinedLoss(nn.Module):
    """
    Combined loss for image reconstruction, using L1, MSE, and perceptual losses.
    """

    def __init__(self, alpha=0.8, beta=0.2, gamma=0.1):
        """
        Args:
            alpha (float): Weight for L1 loss
            beta (float): Weight for MSE loss
            gamma (float): Weight for perceptual loss
        """
        super().__init__()
        self.alpha = alpha  # Weight for L1 loss
        self.beta = beta    # Weight for MSE loss
        self.gamma = gamma  # Weight for perceptual loss
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted image
            target (torch.Tensor): Target image

        Returns:
            torch.Tensor: Combined loss
        """
        # Compute L1 and MSE losses
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)

        # Compute perceptual loss
        perceptual = self.perceptual_loss(pred, target)

        # Weighted combination
        return self.alpha * l1 + self.beta * mse + self.gamma * perceptual

