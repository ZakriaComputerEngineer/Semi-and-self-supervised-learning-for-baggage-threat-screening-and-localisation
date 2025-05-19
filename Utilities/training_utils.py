
"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Zakria Mehmood
# DoC: 2025.04.15
# email: zakriamehmood2001@gmail.com
-----------------------------------------------------------------------------------
# Description: Different utility functions for training and validation
"""

import os
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim

from .Losses import PerceptualLoss, SSIMLoss, CombinedLoss



def early_stopping(val_losses, patience, logger):
    """
    Implements early stopping based on validation loss.

    Args:
        val_losses (list): List of validation losses for each epoch.
        patience (int): Number of epochs to wait before stopping if no improvement.

    Returns:
        (bool, int): (stop_training_flag, best_epoch)
    """
    if len(val_losses) < patience:
        return False, -1  # Not enough epochs to decide

    min_loss = min(val_losses)
    best_epoch = val_losses.index(min_loss) + 1  # Convert to 1-based index

    # If the best validation loss didn't change in the last `patience` epochs, stop training
    if best_epoch < len(val_losses) - patience:
        logger.info(
            f"Early stopping triggered at epoch {len(val_losses)}. Best epoch: {best_epoch}")
        return True, best_epoch
    return False, best_epoch



def save_best_model(model, optimizer, lr_scheduler, epoch, best_loss, val_loss, model_type, logger, configs):
    """
    Saves the best model checkpoint based on validation loss and deletes previous models.

    Args:
        model (torch.nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        lr_scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        epoch (int): Current epoch number.
        best_loss (float): The best validation loss recorded.
        val_loss (float): The current validation loss.
        configs (Namespace): Configuration parameters.
    """
    if val_loss < best_loss:
        best_loss = val_loss
        model_state_dict, utils_state_dict = get_saved_state(
            model, optimizer, lr_scheduler, epoch, configs
        )

        model_save_path = os.path.join(
            configs.model_dir, f'Model_{configs.saved_fn}_epoch_{epoch}_'+model_type+'.pth')
        utils_save_path = os.path.join(
            configs.model_dir, f'Utils_{configs.saved_fn}_epoch_{epoch}_'+model_type+'.pth')

        # Save the new best model
        torch.save(model_state_dict, model_save_path)
        torch.save(utils_state_dict, utils_save_path)
        logger.info(
            f"New best model saved at epoch {epoch} with val_loss: {val_loss:.4f}")

        # Delete previous models to save space
        for file in os.listdir(configs.model_dir):
            if file.startswith('Model_') and file != os.path.basename(model_save_path):
                os.remove(os.path.join(configs.model_dir, file))
            if file.startswith('Utils_') and file != os.path.basename(utils_save_path):
                os.remove(os.path.join(configs.model_dir, file))
                logger.info(f"Deleted previous model checkpoint: {file}")

    return best_loss



def get_saved_state(model, optimizer, lr_scheduler, epoch, configs):
    """Get the information to save with checkpoints"""
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    utils_state_dict = {
        'epoch': epoch,
        'configs': configs,
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'lr_scheduler': copy.deepcopy(lr_scheduler.state_dict())
    }

    return model_state_dict, utils_state_dict



def save_checkpoint(checkpoints_dir, saved_fn, model_state_dict, utils_state_dict, epoch, model_type):
    """Save checkpoint every epoch only is best model or after every checkpoint_freq epoch"""

    model_save_path = os.path.join(
        checkpoints_dir, f'Model_{saved_fn}_epoch_{epoch}_'+model_type+'.pth')
    utils_save_path = os.path.join(
        checkpoints_dir, f'Utils_{saved_fn}_epoch_{epoch}_'+model_type+'.pth')

    torch.save(model_state_dict, model_save_path)
    torch.save(utils_state_dict, utils_save_path)

    print('save a checkpoint at {}'.format(model_save_path))



def get_loss_function(loss_name, smooth=1e-6):
    """
    Get a loss function by name.

    Args:
        loss_name (str): Name of the loss function
        smooth (float): Smoothing factor for dice loss

    Returns:
        callable: Loss function
    """

    if loss_name == 'mse':
        return nn.MSELoss()
    
    elif loss_name == 'l1':
        return nn.L1Loss()
    
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
        
    elif loss_name == 'perceptual':
        return PerceptualLoss()
        
    elif loss_name == 'ssim':
        return SSIMLoss()

    # Special cases that need custom handling
    
    elif loss_name == 'dice':
        
        def dice_loss(pred, target, smooth=smooth):
            pred = torch.sigmoid(pred)
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            intersection = (pred_flat * target_flat).sum()
            return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))
        
        return dice_loss
    
    
    elif loss_name == 'combined':
        return CombinedLoss(alpha=0.8, beta=0.2, gamma=0.1)
    
    
    elif loss_name == 'l1_ssim':
        # Combination of L1 and SSIM
        l1_loss = nn.L1Loss()
        ssim_loss = SSIMLoss()
        def l1_ssim_loss(pred, target):
            return 0.5 * l1_loss(pred, target) + 0.5 * ssim_loss(pred, target)
        return l1_ssim_loss
    
    
    else:
        print(
            f"Unknown loss function: {loss_name}, using combined loss instead")
        return CombinedLoss(alpha=0.8, beta=0.2, gamma=0.1)



def get_activation_function(activation_name):
    """
    Get an activation function by name.

    Args:
        activation_name (str): Name of the activation function

    Returns:
        nn.Module: Activation function
    """
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "silu": nn.SiLU(),  # Added SiLU/Swish activation
        "mish": nn.Mish(),  # Added Mish activation
    }
    return activations.get(activation_name.lower(), nn.ReLU())


def get_optimizer(optimizer_name, model_params, learning_rate=1e-4, weight_decay=1e-5, momentum=0.9):
    """
    Get an optimizer by name.

    Args:
        optimizer_name (str): Name of the optimizer
        model_params (iterable): Model parameters to optimize
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay factor
        momentum (float): Momentum factor for SGD

    Returns:
        torch.optim.Optimizer: Optimizer
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        return optim.SGD(model_params, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == "adam":
        return optim.Adam(model_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(model_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adagrad":
        return optim.Adagrad(model_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        print(f"Unknown optimizer: {optimizer_name}, using AdamW instead")
        return optim.AdamW(model_params, lr=learning_rate, weight_decay=weight_decay)


def create_optimizer(configs, model):
    """
    Create an optimizer for training a transformer-based model with parameter groups.

    This function splits the model parameters into two groups:
      1. Parameters that will receive weight decay (all except biases and normalization parameters).
      2. Parameters that will not receive weight decay (biases and normalization parameters).

    Args:
        configs: Configuration object with optimizer settings
        model (nn.Module): The model to optimize

    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    # Retrieve the model parameters (supporting models wrapped in DistributedDataParallel)
    if hasattr(model, 'module'):
        params_dict = dict(model.module.named_parameters())
    else:
        params_dict = dict(model.named_parameters())

    # Define keywords to exclude from weight decay
    no_decay_keywords = ["bias", "norm",
                         "LayerNorm.weight", "layer_norm.weight"]

    # Split parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for k, v in params_dict.items():
        if any(nd in k for nd in no_decay_keywords):
            no_decay_params.append(v)
        else:
            decay_params.append(v)

    # Create parameter groups
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": configs.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Get optimizer settings
    optimizer_name = configs.optimizer.lower()
    learning_rate = configs.lr

    # Create the optimizer
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=getattr(configs, 'momentum', 0.9),
            nesterov=True
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(
            optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(
            optimizer_grouped_parameters, lr=learning_rate)
    else:
        print(f"Unknown optimizer: {optimizer_name}, using AdamW instead")
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Print information about parameter groups
    print(f'Optimizer: {optimizer_name}, Learning rate: {learning_rate}')
    print(
        f'Parameter groups: {len(decay_params)} parameters with weight decay, {len(no_decay_params)} parameters without weight decay')

    return optimizer


def create_lr_scheduler(optimizer, configs):
    """
    Create a learning rate scheduler based on configuration.

    Args:
        optimizer: The optimizer to schedule
        configs: Configuration object with scheduler settings

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
    """
    scheduler_name = configs.lr_type.lower()

    # Handle warmup if specified
    if hasattr(configs, 'warmup_epochs') and configs.warmup_epochs > 0:
        # Create warmup scheduler
        def warmup_lambda(epoch):
            if epoch < configs.warmup_epochs:
                return float(epoch) / float(max(1, configs.warmup_epochs))
            return 1.0

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda)
        print(f"Using {configs.warmup_epochs} epochs of warmup")
    else:
        warmup_scheduler = None

    # Create main scheduler
    if scheduler_name == 'multi_step':
        def burnin_schedule(i):
            if i < configs.burn_in:
                factor = pow(i / configs.burn_in, 4)
            elif i < configs.steps[0]:
                factor = 1.0
            elif i < configs.steps[1]:
                factor = 0.1
            else:
                factor = 0.01
            return factor

        main_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, burnin_schedule)

    elif scheduler_name == 'cosin':
        # Cosine annealing with warmup
        def cosine_lambda(epoch):
            if warmup_scheduler is not None and epoch < configs.warmup_epochs:
                return warmup_scheduler.get_lr()[0] / configs.lr

            # Cosine decay from https://arxiv.org/pdf/1812.01187.pdf
            adjusted_epoch = epoch - \
                (configs.warmup_epochs if warmup_scheduler else 0)
            adjusted_total = configs.num_epochs - \
                (configs.warmup_epochs if warmup_scheduler else 0)
            return ((1 + math.cos(adjusted_epoch * math.pi / adjusted_total)) / 2) ** 1.0 * 0.9 + 0.1

        main_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=cosine_lambda)

    elif scheduler_name == 'step_lr':
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=configs.lr_step_size, gamma=configs.lr_gamma)

    elif scheduler_name == 'multistep_lr':
        main_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=configs.milestones, gamma=configs.lr_gamma)

    elif scheduler_name == 'exponential_lr':
        main_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=configs.e_gamma)

    elif scheduler_name == 'cosine_annealing_lr':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.t_max, eta_min=configs.min_lr)

    elif scheduler_name == 'reduce_lr_on_plateau':
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=configs.lr_gamma, patience=configs.patience, min_lr=configs.min_lr)

    elif scheduler_name == 'cyclic_lr':
        main_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=configs.base_lr, max_lr=configs.max_lr,
            step_size_up=configs.step_size_up, mode=configs.cyclic_mode)
    else:
        print(
            f"Unknown scheduler: {scheduler_name}, using CosineAnnealingLR instead")
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.num_epochs, eta_min=getattr(configs, 'min_lr', 1e-6))

    # If using warmup, return a SequentialLR
    if warmup_scheduler is not None and scheduler_name != 'cosin':
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[configs.warmup_epochs]
        )

    return main_scheduler