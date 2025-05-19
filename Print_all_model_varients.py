import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary
import sys
import os

# Get the current file's directory and add project root to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Get parent directory
sys.path.append(current_dir)

from Configs.model_configs import MODEL_CONFIGS
from Configs.configs import get_configs
from NN.model_utils import create_model


def main():
    configs = get_configs()

    # Iterate through all model configurations
    for model_name, model_config in MODEL_CONFIGS.items():
        print(f"\nTesting {model_name}:")
        print("-" * 150)

        model, _ = create_model(configs, model=model_name)

        # Print model summary with torchinfo
        # Assuming batch size of 1 and image size from configs
        batch_size = 1
        input_size = (batch_size, configs.in_channels,
                      configs.image_size, configs.image_size)

        print("\nModel Summary:")
        print("-" * 150)
        model_summary = summary(model,
                                input_size=input_size,
                                col_names=["input_size", "output_size",
                                           "num_params", "kernel_size", "mult_adds"],
                                depth=1000,  # Increased depth to show more layers
                                verbose=True  # Show all details
                            )
        print("-" * 150)


if __name__ == "__main__":
    main()
