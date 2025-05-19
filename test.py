import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary

from Utilities.visualize import visualize_reconstructions_2, visualize_segmentation_2
from Utilities.extra import *
from Utilities.log import Logger
from Configs.configs  import get_configs
from NN.model_utils import create_model, make_data_parallel
from Utilities.training_utils import *
from Dataset.Dataloader import DatasetLoader
from Utilities.metrics import calculate_metrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




def main():
    
    configs = get_configs()  # Using our previously defined get_configs()

    # Create necessary directories
    try:
        os.makedirs(configs.logs_dir, exist_ok=True)
        os.makedirs(configs.results_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
        sys.exit(1)

    # Initialize logging
    try:
        logger = Logger(configs.logs_dir, configs.saved_fn)
    except Exception as e:
        print(f"Error initializing logger or TensorBoard writer: {e}")
        sys.exit(1)

    # log the configurations
    try:
        logger.info("Configurations:")
        for key, value in vars(configs).items():
            logger.info(f"{key}: {value}")
    except Exception as e:
        logger.info(f"Error logging configurations: {e}")
        sys.exit(1)

    # Create model and loss function
    try:
        model, loss_fn = create_model(configs, logger, configs.model_name)
    except Exception as e:
        logger.info(f"Error creating model: {e}")
        sys.exit(1)

    # Load pretrained weights if specified
    if configs.pretrained_path is not None:
        try:
            checkpoint = torch.load(configs.pretrained_path, map_location=configs.device)
            # Remove 'module.' from keys if present
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    k = k[7:]  # remove 'module.'
                new_state_dict[k] = v
            # Load into your model
            model.load_state_dict(new_state_dict)
            #model.load_state_dict(torch.load(configs.pretrained_path))
            logger.info(f'Loaded pretrained model from {configs.pretrained_path}')
        except Exception as e:
            logger.info(f"Failed to load pretrained model from {configs.pretrained_path}. Error: {e}")
            sys.exit(1)


    try:
        model = make_data_parallel(model, configs)
    except Exception as e:
        logger.info(f"Error setting up data parallelism: {e}")
        sys.exit(1)

    # Move model to device and setup parallel processing
    try:
        device = torch.device(configs.device)
        model = model.to(device)
        summary_x = summary(model, input_size=(1, 3, 256, 256))  # (batch_size, channels, height, width)
        logger.info(f"Model moved to device: {device}")
        print(f"\n\nModel summary: {summary_x}\n\n")

    except Exception as e:
        logger.info(f"Error moving model to device: {e}")
        sys.exit(1)

    # Load dataset
    try:
        dataset_loader = DatasetLoader(configs.image_size,
                                        configs.root_dir,
                                        configs.split_ratios,
                                        configs.seed,
                                        configs.shuffle_data,
                                        configs.batch_size,
                                        configs.num_workers,
                                        configs.task_type)
        _, _, test_dataloader = dataset_loader.get_dataloaders()
    except Exception as e:
        logger.info(f"Error loading dataset: {e}")
        sys.exit(1)

    # Setup AMP if enabled
    scaler = GradScaler() if configs.use_amp else None



    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluation phase
        model.eval()
        
        # Initialize metrics storage
        all_metrics = []
        running_val_loss = 0.0
        total_samples = 0
        batch = 0
        
        # Lists to store all predictions and ground truth for overall metrics
        all_predictions = []
        all_masks = []
        
        with torch.no_grad():
            for images, masks in tqdm(test_dataloader, desc="Testing", unit="batch"):
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Ensure masks have the correct shape
                if masks.ndim == 2:
                    masks = masks.unsqueeze(0)  # Add batch dimension
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)  # Add channel dimension
                
                
                # Forward pass
                outputs = model(images)
                
                # Store predictions and masks for overall metrics
                all_predictions.append(outputs.cpu())
                all_masks.append(masks.cpu())
                
                if configs.task_type == 'pretext':
                    visualize_reconstructions_2(outputs, images, configs.results_dir, batch)
                    loss = loss_fn(outputs, images)
                elif configs.task_type == 'segmentation':
                    visualize_segmentation_2(outputs, images, masks, batch, configs.results_dir)
                    
                    # Calculate metrics for current batch
                    batch_metrics = calculate_metrics(outputs, masks)
                    all_metrics.append(batch_metrics)
                    
                    # Calculate loss
                    grayscale_outputs = 0.299 * outputs[:, 0, :, :] + 0.587 * outputs[:, 1, :, :] + 0.114 * outputs[:, 2, :, :]
                    grayscale_outputs = grayscale_outputs.unsqueeze(1)
                    loss = loss_fn(grayscale_outputs, masks)
                
                running_val_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                batch += 1

        # Calculate average loss
        avg_val_loss = running_val_loss / total_samples
        
        # Calculate overall metrics
        if configs.task_type == 'segmentation':
            # Concatenate all predictions and masks
            all_predictions = torch.cat(all_predictions, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            
            # Calculate overall metrics
            overall_metrics = calculate_metrics(all_predictions, all_masks)
            
            # Calculate mean metrics across all batches
            mean_metrics = {
                'iou': np.mean([m['iou'] for m in all_metrics]),
                'dice': np.mean([m['dice'] for m in all_metrics]),
                'precision': np.mean([m['precision'] for m in all_metrics]),
                'recall': np.mean([m['recall'] for m in all_metrics]),
                'f1': np.mean([m['f1'] for m in all_metrics])
            }
            
            # Log results
            logger.info(f"\nTest Results:")
            logger.info(f"Average Loss: {avg_val_loss:.4f}")
            logger.info("\nBatch-wise Mean Metrics:")
            for metric, value in mean_metrics.items():
                logger.info(f"{metric.upper()}: {value:.4f}")
            
            logger.info("\nOverall Metrics:")
            for metric, value in overall_metrics.items():
                logger.info(f"{metric.upper()}: {value:.4f}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        cleanup()
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user. Cleaning up...")
        cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        cleanup()
        sys.exit(1)