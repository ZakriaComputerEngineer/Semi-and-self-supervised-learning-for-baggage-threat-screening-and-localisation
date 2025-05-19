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
from torchinfo import summary
from tqdm import tqdm

from Utilities.visualize import visualize_reconstructions, visualize_segmentation
from Utilities.extra import *
from Utilities.log import Logger
from Configs.configs  import get_configs
from NN.model_utils import create_model, make_data_parallel
from Utilities.training_utils import *
from Dataset.Dataloader import DatasetLoader


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer, device, scaler, loss_fn):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    grad_norm = AverageMeter('GradNorm', ':.4e')

    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses, grad_norm],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(
        f"Epoch {epoch+1}/{configs.num_epochs}: Learning rate = {current_lr:.8f}")

    num_iters_per_epoch = len(train_dataloader)

    # switch to train mode
    model.train()
    start_time = time.time()

    for batch_idx, (imgs, masks) in enumerate(train_dataloader):
        # Move data to the correct device
        imgs = imgs.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(imgs)

        # Handle the case where outputs is a tuple (for method4)
        if configs.task_type == 'pretext':
            if isinstance(outputs, tuple):
                # Use the reconstructed image for visualization
                outputs = outputs[0]

        outputs = outputs.to(device)

        # Calculate loss
        if configs.task_type == 'pretext':
            total_loss = loss_fn(outputs, imgs)
        else:
            grayscale_outputs = 0.299 * outputs[:, 0, :, :] + 0.587 * outputs[:, 1, :, :] + 0.114 * outputs[:, 2, :, :]
            grayscale_outputs = grayscale_outputs.unsqueeze(1)  # shape: (batch_size, 1, H, W)
            masks = masks.unsqueeze(1)
            total_loss = loss_fn(grayscale_outputs, masks)
            #total_loss = loss_fn(outputs, masks)

        optimizer.zero_grad()

        if scaler:
            scaler.scale(total_loss).backward()

            # Unscale the gradients for gradient clipping
            scaler.unscale_(optimizer)

            # Compute gradient norm for monitoring
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norm.update(total_norm)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), configs.clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()

            # Compute gradient norm for monitoring
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norm.update(total_norm)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), configs.clip_grad_norm)

            optimizer.step()

        #if batch_idx % configs.subdivisions:
        #    optimizer.zero_grad()

        reduced_loss = total_loss.data

        losses.update(to_python_float(reduced_loss), imgs.size(0))

        # measure elapsed time
        # torch.cuda.synchronize()
        batch_time.update(time.time() - start_time)

        if tb_writer is not None:
            tb_writer.add_scalar('avg_loss', losses.avg, batch_idx)
            tb_writer.add_scalar('Loss/train', losses.avg, epoch)
            tb_writer.add_scalar('GradNorm', grad_norm.avg, batch_idx)

        # Log message
        if logger is not None and batch_idx % 10 == 0:
            logger.info(progress.get_message(batch_idx))

        # More detailed logging every 50 batches
        if logger is not None and batch_idx % 50 == 0:
            logger.info(
                f"Epoch {epoch+1} - Batch {batch_idx}/{len(train_dataloader)} - "
                f"Loss: {total_loss:.4f}, Grad Norm: {grad_norm.val:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}")

        start_time = time.time()

    # Visualize outputs every few epochs
    #if epoch % 5 == 0 and configs.task_type == 'pretext':
    #    visualize_reconstructions(
    #        model, train_dataloader, device, epoch, configs.results_dir)
    #elif configs.task_type == 'segmentation':
    #    visualize_segmentation(
    #        model, train_dataloader, device, epoch, configs.results_dir)


def main():
    try:
        configs = get_configs()  # Using our previously defined get_configs()

        # Create necessary directories
        try:
            os.makedirs(configs.checkpoints_dir, exist_ok=True)
            os.makedirs(configs.logs_dir, exist_ok=True)
            os.makedirs(configs.results_dir, exist_ok=True)
            os.makedirs(configs.model_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directories: {e}")
            sys.exit(1)

        # Set seeds for reproducibility
        if configs.seed is not None:
            random.seed(configs.seed)
            np.random.seed(configs.seed)
            torch.manual_seed(configs.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Initialize logging
        try:
            logger = Logger(configs.logs_dir, configs.saved_fn)
            writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))
        except Exception as e:
            print(f"Error initializing logger or TensorBoard writer: {e}")
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

        # Create optimizer and scheduler
        try:
            optimizer = create_optimizer(configs, model)
            lr_scheduler = create_lr_scheduler(optimizer, configs)
        except Exception as e:
            logger.info(f"Error creating optimizer or learning rate scheduler: {e}")
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
                                           configs.task_type,)
            train_dataloader, val_dataloader, _ = dataset_loader.get_dataloaders()
        except Exception as e:
            logger.info(f"Error loading dataset: {e}")
            sys.exit(1)

        # Setup AMP if enabled
        scaler = GradScaler() if configs.use_amp else None

        # Training loop
        best_val_loss = float("inf")
        val_losses = []

        for epoch in range(configs.start_epoch, configs.num_epochs):
            try:
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Train for one epoch
                train_one_epoch(train_dataloader, model, optimizer,
                                lr_scheduler, epoch, configs, logger, writer,
                                device, scaler, loss_fn)

                # Validation phase
                model.eval()
                running_val_loss = 0.0

                with torch.no_grad():
                    for images, masks in tqdm(val_dataloader, desc=f"Validation Epoch {epoch}"):
                        images = images.to(device)
                        masks = masks.to(device)

                        with autocast() if scaler else torch.no_grad():
                            outputs = model(images)

                            if configs.task_type == 'pretext':
                                loss = loss_fn(outputs, images)
                            else:
                                grayscale_outputs = 0.299 * outputs[:, 0, :, :] + 0.587 * outputs[:, 1, :, :] + 0.114 * outputs[:, 2, :, :]
                                grayscale_outputs = grayscale_outputs.unsqueeze(1)  # shape: (batch_size, 1, H, W)
                                masks = masks.unsqueeze(1)
                                
                                loss = loss_fn(grayscale_outputs, masks)

#                            loss = loss_fn(outputs, images if configs.task_type == "pretext" else masks)


                        running_val_loss += loss.item()

                avg_val_loss = running_val_loss / len(val_dataloader)
                val_losses.append(avg_val_loss)
                writer.add_scalar('Loss/val', avg_val_loss, epoch)

                # Save best model
                if epoch % configs.save_checkpoint_freq == 0:
                    save_checkpoint(configs.checkpoints_dir,
                                    configs.saved_fn + '_best',
                                    model.state_dict(),
                                    {'optimizer': optimizer.state_dict(),
                                     'lr_scheduler': lr_scheduler.state_dict(),
                                     'epoch': epoch},
                                    epoch, model_type=configs.task_type)

                # Early stopping check
                if len(val_losses) > configs.patience:
                    if all(val_losses[-i-1] >= val_losses[-i-2]
                           for i in range(configs.patience)):
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break

                # Log validation loss
                logger.info(
                    f"Validation Loss: {avg_val_loss:.4f} at epoch {epoch+1}/{configs.num_epochs}")
                

                if epoch % 5 == 0:
                    if configs.task_type == 'pretext':
                        visualize_reconstructions(model, train_dataloader,
                                                device, epoch, configs.results_dir)
                    elif configs.task_type == 'segmentation':
                        visualize_segmentation(model, train_dataloader,
                                            device, epoch, configs.results_dir)

            except Exception as e:
                logger.info(f"Error during training or validation at epoch {epoch}: {e}")
                cleanup()
                sys.exit(1)

        writer.close()

    except KeyboardInterrupt:
        print("Training interrupted by user. Cleaning up...")
        cleanup()
        sys.exit(0)
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