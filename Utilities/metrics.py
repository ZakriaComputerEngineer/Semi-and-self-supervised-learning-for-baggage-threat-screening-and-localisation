import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import copy 
import matplotlib.pyplot as plt

def calculate_iou(preds, targets, num_classes=1):
    preds = preds.argmax(dim=1)  # Get the predicted class
    iou_list = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (targets == cls)).sum().item()
        union = ((preds == cls) | (targets == cls)).sum().item()
        iou = intersection / (union + 1e-6)  # Avoid division by zero
        iou_list.append(iou)
    return sum(iou_list) / len(iou_list) if iou_list else 0

def calculate_metrics(outputs, masks):
    """
    Calculate performance metrics for segmentation.
    
    Args:
        outputs: Model predictions
        masks: Ground truth masks
        
    Returns:
        dict: Dictionary containing various metrics
    """
    # Ensure inputs are properly shaped
    if outputs.ndim == 3:
        outputs = outputs.unsqueeze(1)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
        
    # Convert outputs to grayscale and apply sigmoid
    grayscale_outputs = 0.299 * outputs[:, 0, :, :] + 0.587 * outputs[:, 1, :, :] + 0.114 * outputs[:, 2, :, :]
    grayscale_outputs = grayscale_outputs.unsqueeze(1)
    predictions = torch.sigmoid(grayscale_outputs)
    
    # Threshold predictions to get binary mask
    predictions = (predictions > 0.5).float()
    
    # Ensure masks are float tensors
    masks = masks.float()
    
    # Ensure predictions and masks have the same shape
    if predictions.shape != masks.shape:
        predictions = predictions.squeeze(1)
        masks = masks.squeeze(1)
    
    # Calculate intersection and union using float operations
    intersection = (predictions * masks).sum((1, 2) if predictions.ndim == 3 else (1, 2, 3))
    union = (predictions + masks).clamp(0, 1).sum((1, 2) if predictions.ndim == 3 else (1, 2, 3))
    
    # IoU (Jaccard)
    iou = (intersection + 1e-6) / (union + 1e-6)
    mean_iou = iou.mean().item()
    
    # Dice coefficient
    dice = (2 * intersection + 1e-6) / (predictions.sum((1, 2) if predictions.ndim == 3 else (1, 2, 3)) + masks.sum((1, 2) if predictions.ndim == 3 else (1, 2, 3)) + 1e-6)
    mean_dice = dice.mean().item()
    
    # Precision and Recall
    true_positives = intersection
    false_positives = predictions.sum((1, 2) if predictions.ndim == 3 else (1, 2, 3)) - intersection
    false_negatives = masks.sum((1, 2) if predictions.ndim == 3 else (1, 2, 3)) - intersection
    
    precision = (true_positives + 1e-6) / (true_positives + false_positives + 1e-6)
    recall = (true_positives + 1e-6) / (true_positives + false_negatives + 1e-6)
    
    mean_precision = precision.mean().item()
    mean_recall = recall.mean().item()
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    mean_f1 = f1.mean().item()
    
    return {
        'iou': mean_iou,
        'dice': mean_dice,
        'precision': mean_precision,
        'recall': mean_recall,
        'f1': mean_f1
    }


def calculate_dice(preds, targets):
    preds = preds.argmax(dim=1)  # Get the predicted class
    intersection = (preds & targets).sum().item()
    return (2. * intersection) / (preds.sum().item() + targets.sum().item() + 1e-6)


def calculate_mse(preds, targets):
    return F.mse_loss(preds, targets)


def calculate_psnr(mse):
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def plot_lr_scheduler(optimizer, scheduler, num_epochs=300, save_dir=''):
    # Plot LR simulating training for full num_epochs
    optimizer, scheduler = copy.copy(optimizer), copy.copy(
        scheduler)  # do not modify originals
    y = []
    for _ in range(num_epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, num_epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'LR.png'), dpi=200)


