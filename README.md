# Semi-and-self-supervised-learning-for-baggage-threat-screening-and-localisation

*A PyTorch implementation of a Vision Transformer (ViT) Autoencoder for semi and self-supervised learning applied to baggage threat screening and localization.*

![whole_architecture](https://github.com/user-attachments/assets/b005d87c-31ba-41af-b743-df2a0f05c691)

---

## Overview

This repository implements a full training pipeline for **semi-supervised and self-supervised learning** approaches in the context of **X-ray baggage threat segmentation** using ViT-based architectures. It supports both pretext (reconstruction) and downstream (segmentation) tasks and is designed for **reproducibility** and **research extensibility**.

  - upstream pretext task: pretraining via image reconstruction
  - Downstream segmentation task: training by segmenting mask for threat localization

---

## Features

Vision Transformer Architecture
  - Patch embedding
  - Multi-head self-attention
  - Encoder-decoder transformer structure
  - Multiple reconstruction methods

Training Approaches
  - Pretext task training (self-supervised)
  - Supervised segmentation (you can also train it as a simple VIT-AE for your data
  - Transfer learning from pretext to downstream tasks

Advanced Training Features
  - Mixed precision training
  - Gradient clipping
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing
  - Distributed training support

---

## Code Structure

````
├── Complete_Notebook.ipynb     # Main implementation notebook
├── Print_all_model_varients.py
├── requirements.txt
├── test.py                     # Test script
├── train.py                    # Training script
├── Configs/
│   ├── configs.py             # Configuration files
│   └── model_configs.py
│   └── arg_parser.py
├── Dataset/
│   └── Dataloader.py          # Data loading utilities
├── NN/
│   ├── model.py               # Model architecture
│   ├── model_utils.py         # Model utilities
│   └── Layers/
└── Utilities/
    ├── extra.py               # Additional utilities
    ├── log.py                 # Logging utilities
    ├── Losses.py             # Loss functions
    ├── metrics.py            # Evaluation metrics
    ├── training_utils.py     # Training helpers
    └── visualize.py          # Visualization tools
````
---

## Requirements

- torch
- torchvision
- numpy
- pillow
- einops
- tqdm
- easydict
- torchinfo
- matplotlib
- opencv-python

---

## Data structure

````
├── root_directory\
    ├── dataset_1\
        ├── images\
        ├── masks\
    ├── dataset_2\
        ├── images\
        ├── masks\
    ├── dataset_3\
        ├── images\
        ├── masks\
    .
    .
    .
````
## Data Augmentation present in dataloader class

- 4 axis rotation [45,135,225,315]
- flips [horizontal, vertical]
- scales [zoom in, zoom out]
- shifts [right, left]

Constants:
- resolution: 768x576
- cv2 copy border
- mask normalization

---

## Options

1) Loss function
2) Optimizer
3) Reconstruction/Segmentation head
4) activation

*please go through the arg_parser file and select desired type, reconstruction/segmentation head is selection is available in model_configs file!*

---

## Training commands

- for upstream:
```python train.py --task_type pretext --model model_tiny/model_small/model_base/model_large/model_huge --root_dir "path to your data root directory"```

- for downstream:
```python train.py --task_type segmentation --pretrained_path "path to your upstream trained model --root_dir "path to your data root directory""```

---

## Testing commands

```python test.py --pretrained_path "path to your complete trainined model"  --root_dir "path to your data root directory""```

---

## Results

*Segmentation results*
![collage_results](https://github.com/user-attachments/assets/64eb3605-94a8-4d0e-9598-fca695ed1d9f)

*Reconstruction results*
![recon_result](https://github.com/user-attachments/assets/4dfcba34-408e-45bd-a9e5-3bfdee8ea0bd)

*Data Augmentation sample*
![image_augmentation](https://github.com/user-attachments/assets/be9bcfc0-3efe-47b2-89d2-dd3e6063b6a7)

*Training curve*
![train_val_self-supervised](https://github.com/user-attachments/assets/68ead744-9c60-4bc8-b3a7-ff89883d616f)

---

## Performance Comparison (supervised VS self-supervised)

| Dataset | IOU (Self-Supervised) | F1-Score (Self-Supervised) | Precision (Self-Supervised) | Recall (Self-Supervised) | IOU (Supervised) | F1-Score (Supervised) | Precision (Supervised) | Recall (Supervised) |
|---------|------------------------|-----------------------------|------------------------------|---------------------------|------------------|------------------------|-------------------------|----------------------|
| GDXRAY  | 0.9213                 | 0.9166                      | 0.9016                       | 0.9204                    | 0.8243           | 0.8338                 | 0.8141                  | 0.8051               |
| SIXRAY  | 0.9014                 | 0.9174                      | 0.9010                       | 0.9055                    | 0.7391           | 0.8035                 | 0.8075                  | 0.8188               |
| PIDRAY  | 0.8927                 | 0.8835                      | 0.9004                       | 0.8919                    | 0.7216           | 0.8116                 | 0.8277                  | 0.8066               |

---

## Acknowledgements
*This work was conducted at the National University of Sciences & Technology (NUST), Islamabad, Pakistan under the guidance of Dr. Usman Akram.*

---
