import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary

from Configs.model_configs import MODEL_CONFIGS
from .model import VisionTransformerAutoencoder
from Utilities.training_utils import get_loss_function

def initialize_weights(model):
    """
    Initialize all model weights using Xavier uniform initialization.
    
    Args:
        model (nn.Module): The model to initialize
        
    Returns:
        model (nn.Module): The initialized model
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Xavier uniform for all linear layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            # Standard initialization for normalization layers
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            
        elif isinstance(m, nn.Conv2d):
            # Xavier uniform for all convolutional layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, nn.ConvTranspose2d):
            # Xavier uniform for transpose convolutions
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, nn.MultiheadAttention):
            # Xavier uniform for attention layers
            if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight)
            
            # Initialize out projection
            if hasattr(m, 'out_proj') and hasattr(m.out_proj, 'weight'):
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
            
            # Initialize separate Q, K, V projections if they exist
            for weight_name in ['q_proj_weight', 'k_proj_weight', 'v_proj_weight']:
                if hasattr(m, weight_name) and getattr(m, weight_name) is not None:
                    weight = getattr(m, weight_name)
                    nn.init.xavier_uniform_(weight)

    # Initialize positional embeddings if they exist
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'pos_embed'):
        nn.init.xavier_uniform_(model.patch_embed.pos_embed)
    
    # Initialize decoder positional embeddings if they exist
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'decoder_pos_embed'):
        nn.init.xavier_uniform_(model.decoder.decoder_pos_embed)

    return model


def verify_initialization(model):
    """
    Verify that the model weights are properly initialized using Xavier uniform initialization.
    
    Args:
        model: The model to verify
        
    Returns:
        bool: True if initialization is correct, False otherwise
    """
    initialization_ok = True
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'weight' in name and param.dim() >= 2:
                # Calculate expected Xavier bounds
                fan_in = param.size(1)
                fan_out = param.size(0)
                std = math.sqrt(2.0 / (fan_in + fan_out))
                bound = math.sqrt(3.0) * std
                
                # Check if weights are within Xavier bounds
                if torch.any(torch.abs(param) > bound):
                    print(f"Warning: {name} has weights outside Xavier bounds (-{bound:.4f}, {bound:.4f})")
                    initialization_ok = False
                
                # Check weight distribution
                actual_std = torch.std(param).item()
                expected_std = std
                if not (0.7 * expected_std <= actual_std <= 1.3 * expected_std):
                    print(f"Warning: {name} has unexpected standard deviation: {actual_std:.4f} (expected ≈ {expected_std:.4f})")
                    initialization_ok = False
                
                # Check for dead neurons
                if torch.any(torch.all(param == 0, dim=1)):
                    print(f"Warning: {name} has dead neurons (all-zero weights)")
                    initialization_ok = False
            
            # Check normalization layers
            elif any(x in name for x in ['norm', 'ln', 'batch_norm']) and 'weight' in name:
                if not torch.allclose(param, torch.ones_like(param), rtol=1e-3):
                    print(f"Warning: {name} normalization weights not initialized to ones")
                    initialization_ok = False
            
            # Verify bias initialization
            elif 'bias' in name:
                if not torch.allclose(param, torch.zeros_like(param), rtol=1e-3):
                    print(f"Warning: {name} bias not initialized to zeros")
                    initialization_ok = False
    
    # Verify positional embeddings
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'pos_embed'):
        pos_embed = model.patch_embed.pos_embed
        fan_in = pos_embed.size(-1)
        fan_out = pos_embed.size(-2)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        bound = math.sqrt(3.0) * std
        
        if torch.any(torch.abs(pos_embed) > bound):
            print(f"Warning: Positional embeddings outside Xavier bounds (-{bound:.4f}, {bound:.4f})")
            initialization_ok = False
    
    # Verify decoder positional embeddings
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'decoder_pos_embed'):
        dec_pos_embed = model.decoder.decoder_pos_embed
        fan_in = dec_pos_embed.size(-1)
        fan_out = dec_pos_embed.size(-2)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        bound = math.sqrt(3.0) * std
        
        if torch.any(torch.abs(dec_pos_embed) > bound):
            print(f"Warning: Decoder positional embeddings outside Xavier bounds (-{bound:.4f}, {bound:.4f})")
            initialization_ok = False
    
    if initialization_ok:
        print("All weights properly initialized with Xavier uniform distribution!")
        print(f"Note: Weights are bounded by their respective fan-in/fan-out values")
    
    return initialization_ok



def log_initialization(model, logger=None):
    """
    Log information about the initialized model.

    Args:
        model (nn.Module): The initialized model
        logger: Logger object to log information
    """
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    # Log initialization info
    if logger:
        logger.info(f"Model initialized with Xavier weights")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Log model structure
        logger.info("Model structure:")
        logger.info(
            f"- Patch Embedding: {model.patch_embed.__class__.__name__}")
        logger.info(
            f"- Encoder: {model.encoder.__class__.__name__} with {len(model.encoder.layers)} layers")
        logger.info(
            f"- Decoder: {model.decoder.__class__.__name__} with {len(model.decoder.decoder_blocks)} layers")
        if hasattr(model.decoder, 'reconstruction'):
            logger.info(
                f"- Reconstruction method: {model.decoder.reconstruction.__class__.__name__}")
    else:
        print(f"Model initialized with Xavier weights")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")





def create_model(configs, logger=None, model=None):
    """
    Create model based on architecture name and return the model and appropriate loss function.

    Args:
        configs: Configuration object with model parameters
        logger: Optional logger for logging information

    Returns:
        tuple: (model, loss_function) - The initialized model and appropriate loss function
    """

    # Check if model architecture is specified
    network = MODEL_CONFIGS[model]

    if logger is not None:
        logger.info(f'Using model: {model}')
        logger.info('Model Network: {}'.format(network))

    # Get reconstruction method from config or use default
    recon_method = network.get('recon_method', 'method1')

    output_channels = 3  # Default output channels for RGB images

    model = VisionTransformerAutoencoder(configs.image_size,
                                            configs.patch_size,
                                            configs.in_channels,
                                            int(network['embed_dim']),
                                            int(network['depth']),
                                            int(network['num_heads']),
                                            int(network['decoder_embed_dim']),
                                            int(network['decoder_depth']),
                                            int(network['decoder_num_heads']),
                                            int(network['mlp_ratio']),
                                            network['activation'],
                                            recon_method,
                                            int(network['dropout']),
                                            output_channels = output_channels)

    # Initialize model weights using Xavier initialization
    model = initialize_weights(model)

    # Log initialization information
    log_initialization(model, logger)

    # Verify initialization
    verify_initialization(model)

    # Create appropriate loss function
    if configs.task_type == 'pretext':
        loss_fn = create_reconstruction_loss(recon_method)
    else:
        loss_fn = get_loss_function(configs.loss_function)

    if logger is not None:
        logger.info(f'Using reconstruction method: {recon_method}')
        logger.info(f'Using loss function: {configs.loss_function}')
        logger.info(f"Model architecture: {model.__class__.__name__}")
        logger.info(f"Total parameters: {get_num_parameters(model):,}")

    return model, loss_fn


def get_num_parameters(model):
    """Count number of trained parameters of the model"""
    if hasattr(model, 'module'):
        num_parameters = sum(p.numel()
                             for p in model.module.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel()
                             for p in model.parameters() if p.requires_grad)

    return num_parameters


def make_data_parallel(model, configs):
    if configs.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if configs.gpu_idx is not None:
            torch.cuda.set_device(configs.gpu_idx)
            model.cuda(configs.gpu_idx)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            configs.batch_size = int(
                configs.batch_size / configs.ngpus_per_node)
            configs.num_workers = int(
                (configs.num_workers + configs.ngpus_per_node - 1) / configs.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[configs.gpu_idx])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif configs.gpu_idx is not None:
        torch.cuda.set_device(configs.gpu_idx)
        model = model.cuda(configs.gpu_idx)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    return model


def create_reconstruction_loss(recon_method="method1"):
    """
    Create a loss function suitable for the specified reconstruction method.

    Args:
        recon_method (str): The reconstruction method being used

    Returns:
        function: A loss function that takes model output and target as input
    """
    if recon_method == "method4":
        # For segmentation-optimized reconstruction (method4)
        def segmentation_pretraining_loss(output, target):
            # Unpack outputs if in training mode
            if isinstance(output, tuple):
                recon_img, semantic_features = output

                # Reconstruction loss (L1 loss for better edge preservation)
                recon_loss = nn.functional.l1_loss(recon_img, target)

                # Feature consistency loss (encourage similar features for similar regions)
                # This helps the model learn semantic representations useful for segmentation
                # We use a simple proxy by computing gradients in the target image
                target_gray = 0.299 * \
                    target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
                target_gray = target_gray.unsqueeze(1)  # Add channel dimension

                # Compute gradients using Sobel filters
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                       dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                       dtype=torch.float32, device=target.device).view(1, 1, 3, 3)

                # Apply filters
                grad_x = nn.functional.conv2d(target_gray, sobel_x, padding=1)
                grad_y = nn.functional.conv2d(target_gray, sobel_y, padding=1)
                grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

                # Normalize gradient magnitude to [0, 1]
                grad_magnitude = grad_magnitude / \
                    (torch.max(grad_magnitude) + 1e-8)

                # Feature consistency loss - higher weight on edge regions
                edge_weight = 1.0 + 5.0 * grad_magnitude  # Higher weight on edges
                feature_loss = torch.mean(edge_weight * torch.abs(
                    nn.functional.normalize(semantic_features, dim=1) -
                    nn.functional.normalize(grad_magnitude, dim=1)
                ))

                # Total loss with weighting
                total_loss = recon_loss + 0.5 * feature_loss
                return total_loss
            else:
                # If not in training mode, just use L1 loss
                return nn.functional.l1_loss(output, target)

        return segmentation_pretraining_loss
    else:
        # For other reconstruction methods, use a combination of L1 and SSIM loss
        def standard_reconstruction_loss(output, target):
            # L1 loss for pixel-wise accuracy
            l1_loss = nn.functional.l1_loss(output, target)

            # MSE loss for overall image similarity
            mse_loss = nn.functional.mse_loss(output, target)

            # Combined loss
            return 0.8 * l1_loss + 0.2 * mse_loss

        return standard_reconstruction_loss

