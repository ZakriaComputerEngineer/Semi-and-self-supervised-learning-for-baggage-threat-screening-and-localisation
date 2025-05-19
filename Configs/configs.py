import torch
from easydict import EasyDict as edict
from .model_configs import MODEL_CONFIGS

def get_configs():
    configs = {
        # Task configuration
        'task_type': 'segmentation',  # pretext (reconstruction) or downstream (segmentation)
        
        # General configs
        'patch_size': 16,
        'model': 'model_base',
        'saved_fn': 'vit_autoencoder',
        'arch': 'vit',
        
        # Dataset configs
        'root_dir': '/home/zakriamehmood@ISB.MTBC.COM/Desktop/STT/another_one/Augmented_Dataset',
        'split_ratios': [0.8, 0.15, 0.05],
        'shuffle_data': True,
        'seed': 42,
        
        # Training configs
        'num_epochs': 100,
        'start_epoch': 0,
        'optimizer': 'adamw',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'loss_function': 'bce', #only works on segmentation task
        'batch_size': 64,
        'num_workers': 4,
        'activation_function': 'gelu',
        'early_stopping_patience': 20,
        'clip_grad_norm': 1.0,
        'use_amp': True,
        'evaluate': False,
        'print_freq': 50,
        'checkpoint_freq': 5,
        'save_best_freq': 1,
        
        # Learning rate configs
        'lr': 5e-5,
        'lr_type': 'cosine',
        'burn_in': 10,
        'steps': [1500, 4000],
        'lr_step_size': 10,
        'lr_gamma': 0.1,
        'weight_decay': 1e-4,
        'minimum_lr': 1e-6,
        'milestones': [30, 60, 90],
        'e_gamma': 0.95,
        't_max': 10,
        'min_lr': 1e-6,
        'base_lr': 1e-6,
        'max_lr': 1e-3,
        'step_size_up': 2000,
        'cyclic_mode': 'triangular',
        
        # Distributed training configs
        'world_size': -1,
        'rank': -1,
        'dist_url': 'tcp://127.0.0.1:29500',
        'dist_backend': 'nccl',
        'gpu_idx': None,
        'no_cuda': False,
        'distributed': False,
        
        # Paths
        'pretrained_path': 'Model_vit_autoencoder_best_epoch_95_segmentation.pth',
        'logs_dir': 'logs',
        'results_dir': 'results',
        'model_dir': 'model',
        
        # New configs
        'early_stopping': True,
        'patience': 15,
        'save_checkpoint_freq': 5,
        'step_lr_in_epoch': True,
        'warmup_epochs': 5,
        'normalize_data': True
    }
    
    # Convert to EasyDict
    configs = edict(configs)
    
    # Update with model specific configs
    if configs.model in MODEL_CONFIGS:
        model_config = MODEL_CONFIGS[configs.model]
        for key, value in model_config.items():
            setattr(configs, key, value)
    
    return configs
