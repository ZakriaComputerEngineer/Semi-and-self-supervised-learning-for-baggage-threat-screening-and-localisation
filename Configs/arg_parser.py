import argparse
import torch
from .model_configs import MODEL_CONFIGS

def get_parser():
    parser = argparse.ArgumentParser(description='Model Configuration')
    
    # Task configuration
    parser.add_argument('--task_type', type=str, default='segmentation',
                      help='pretext (reconstruction) or downstream (segmentation)')
    
    # General configs
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--model', type=str, default='model_base')
    parser.add_argument('--saved_fn', type=str, default='vit_autoencoder')
    parser.add_argument('--arch', type=str, default='vit')
    
    # Dataset configs
    parser.add_argument('--root_dir', type=str, 
                      default='/home/zakriamehmood@ISB.MTBC.COM/Desktop/STT/another_one/Augmented_Dataset')
    parser.add_argument('--split_ratios', nargs=3, type=float, default=[0.8, 0.15, 0.05])
    parser.add_argument('--shuffle_data', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    
    # Training configs
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--loss_function', type=str, default='bce')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--activation_function', type=str, default='gelu')
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--checkpoint_freq', type=int, default=5)
    parser.add_argument('--save_best_freq', type=int, default=1)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--save_checkpoint_freq', type=int, default=5)
    parser.add_argument('--step_lr_in_epoch', type=bool, default=True)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--normalize_data', type=bool, default=True)
    
    # Learning rate configs
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_type', type=str, default='cosine')
    parser.add_argument('--burn_in', type=int, default=10)
    parser.add_argument('--steps', nargs='+', type=int, default=[1500, 4000])
    parser.add_argument('--lr_step_size', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--minimum_lr', type=float, default=1e-6)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 60, 90])
    parser.add_argument('--e_gamma', type=float, default=0.95)
    parser.add_argument('--t_max', type=int, default=10)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--base_lr', type=float, default=1e-6)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--step_size_up', type=int, default=2000)
    parser.add_argument('--cyclic_mode', type=str, default='triangular')
    
    # Distributed training configs
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--rank', type=int, default=-1)
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:29500')
    parser.add_argument('--dist_backend', type=str, default='nccl')
    parser.add_argument('--gpu_idx', type=int, default=None)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--distributed', type=bool, default=False)
    
    # Paths
    parser.add_argument('--pretrained_path', type=str, 
                      default='Model_vit_autoencoder_best_epoch_95_segmentation.pth')
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--model_dir', type=str, default='model')
    
    return parser

def parse_args_to_config(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    
    # Convert args to dictionary
    config_dict = vars(args)
    
    # Update with model specific configs
    if config_dict['model'] in MODEL_CONFIGS:
        config_dict.update(MODEL_CONFIGS[config_dict['model']])
    
    # Add device
    config_dict['device'] = 'cuda' if torch.cuda.is_available() and not config_dict['no_cuda'] else 'cpu'
    
    return config_dict
