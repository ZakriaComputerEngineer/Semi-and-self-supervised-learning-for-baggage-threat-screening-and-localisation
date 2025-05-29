from easydict import EasyDict as edict
from .arg_parser import parse_args_to_config

def get_configs(args=None):
    # Get configs from command line arguments
    configs = parse_args_to_config(args)
    
    # Convert to EasyDict and return
    return edict(configs)
