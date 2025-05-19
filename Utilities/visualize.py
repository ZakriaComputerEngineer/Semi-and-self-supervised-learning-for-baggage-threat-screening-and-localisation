import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np

from .extra import denormalize


def visualize_reconstructions(model, dataloader, device, epoch, save_dir):
    """
    Visualize and save model reconstructions during training
    """
    model.eval()
    with torch.no_grad():
        # Get a batch of images
        images, _ = next(iter(dataloader))
        images = images.to(device)

        # Generate reconstructions
        reconstructions = model(images)
        reconstructions = reconstructions.to(device)
        # Create a grid of original and reconstructed images
        comparison = torch.cat([images[:8], reconstructions[:8]])
        grid = torchvision.utils.make_grid(comparison, nrow=8, normalize=True)

        # Save the grid
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(
            grid, f"{save_dir}/reconstruction_epoch_{epoch}.png")

        # Also save as matplotlib figure for better visualization
        plt.figure(figsize=(20, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Reconstructions at Epoch {epoch}")
        plt.axis('off')
        plt.savefig(f"{save_dir}/reconstruction_plot_epoch_{epoch}.png")
        plt.close()

    model.train()
    return reconstructions




def denormalize(img, mean, std):
    """
    Undo normalization for visualization.
    img: (C, H, W) tensor
    """
    img = img.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = torch.clamp(img, 0, 1)  # Clamp to [0, 1] for safe visualization
    return img

def visualize_segmentation(model, dataloader, device, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(dataloader))
        images = images.to(device)
        masks = masks.to(device)

        # Fix masks shape
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        predictions = model(images)
        grayscale_outputs = 0.299 * predictions[:, 0, :, :] + 0.587 * predictions[:, 1, :, :] + 0.114 * predictions[:, 2, :, :]
        grayscale_outputs = grayscale_outputs.unsqueeze(1)
        predictions = torch.sigmoid(grayscale_outputs)
        predictions = (predictions > 0.5).float()

        # Move tensors to CPU
        images = images.cpu()
        masks = masks.cpu()
        predictions = predictions.cpu()

        # Denormalization parameters (standard ImageNet mean/std)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        num_samples = min(8, images.size(0))
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

        for i in range(num_samples):
            # De-normalize input image
            img = denormalize(images[i], mean, std)

            # Input image
            axes[i, 0].imshow(img.permute(1, 2, 0).numpy())
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis("off")

            # Ground truth mask
            axes[i, 1].imshow(masks[i, 0].numpy(), cmap="gray")
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis("off")

            # Predicted mask
            axes[i, 2].imshow(predictions[i, 0].numpy(), cmap="gray")
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis("off")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"segmentation_epoch_{epoch}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Segmentation visualization saved at {save_path}")

    model.train()

def visualize_reconstructions_2(reconstructions, images, save_dir, batch):
    """
    Visualize and save model reconstructions during testing
    """
    
    # Create a grid of original and reconstructed images
    comparison = torch.cat([images[:8], reconstructions[:8]])
    grid = torchvision.utils.make_grid(comparison, nrow=8, normalize=True)

    torchvision.utils.save_image(
        grid, f"{save_dir}/reconstruction_results_{batch}.png")

    # Also save as matplotlib figure for better visualization
    plt.figure(figsize=(20, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Reconstructions of test batch {batch}")
    plt.axis('off')
    plt.savefig(f"{save_dir}/reconstruction_results_{batch}.png")
    plt.close()


def visualize_segmentation_2(predictions, images, masks, batch, save_dir):
    """
    Visualize and save segmentation results during testing with error handling.
    """
    try:
        # Fix masks shape
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        grayscale_outputs = 0.299 * predictions[:, 0, :, :] + 0.587 * predictions[:, 1, :, :] + 0.114 * predictions[:, 2, :, :]
        grayscale_outputs = grayscale_outputs.unsqueeze(1)
        predictions = torch.sigmoid(grayscale_outputs)
        predictions = (predictions > 0.5).float()

        # Move tensors to CPU
        try:
            images = images.cpu()
            masks = masks.cpu()
            predictions = predictions.cpu()
        except Exception as e:
            print(f"Error moving tensors to CPU: {e}")
            return

        # Denormalization parameters (standard ImageNet mean/std)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        num_samples = min(16, images.size(0))
        try:
            fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        except Exception as e:
            print(f"Error creating subplot figure: {e}")
            return

        try:
            for i in range(num_samples):
                # De-normalize input image
                img = denormalize(images[i], mean, std)

                # Input image
                axes[i, 0].imshow(img.permute(1, 2, 0).numpy())
                axes[i, 0].set_title("Input Image")
                axes[i, 0].axis("off")

                # Ground truth mask
                axes[i, 1].imshow(masks[i, 0].numpy(), cmap="gray")
                axes[i, 1].set_title("Ground Truth Mask")
                axes[i, 1].axis("off")

                # Predicted mask
                axes[i, 2].imshow(predictions[i, 0].numpy(), cmap="gray")
                axes[i, 2].set_title("Predicted Mask")
                axes[i, 2].axis("off")

        except Exception as e:
            print(f"Error plotting images: {e}")
            plt.close(fig)
            return

        try:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"segmentation_result_batch_{batch}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving visualization: {e}")
            plt.close(fig)
            return

    except Exception as e:
        print(f"Unexpected error in visualize_segmentation_2: {e}")
        # Ensure figure is closed in case of error
        try:
            plt.close()
        except:
            pass
        return





