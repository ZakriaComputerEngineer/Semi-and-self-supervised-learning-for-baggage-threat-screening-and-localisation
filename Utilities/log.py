"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Zakria Mehmood
# DoC: 2025.04.15
# email: zakriamehmood2001@gmail.com
-----------------------------------------------------------------------------------
# Description: Logging class
"""

import logging
import os

class Logger():
    """
        Create logger to save logs during training
        Args:
            logs_dir:
            saved_fn:

        Returns:

        """

    def __init__(self, logs_dir, saved_fn):
        logger_fn = 'logger_{}.txt'.format(saved_fn)
        logger_path = os.path.join(logs_dir, logger_fn)
#/content/Self Supervised Learning - Copy/logs/logger_vit_autoencoder.txt
        with open(logger_path, "w") as file:
          pass


        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # formatter = logging.Formatter('%(asctime)s:File %(module)s.py:Func %(funcName)s:Line %(lineno)d:%(levelname)s: %(message)s')
        formatter = logging.Formatter(
            '%(asctime)s: %(module)s.py - %(funcName)s(), at Line %(lineno)d:%(levelname)s:\n%(message)s')

        file_handler = logging.FileHandler(logger_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def info(self, message):
        self.logger.info(message)



def get_tensorboard_log(model):
    if hasattr(model, 'module'):
        yolo_layers = model.module.yolo_layers
    else:
        yolo_layers = model.yolo_layers

    tensorboard_log = {}
    tensorboard_log['Average_All_Layers'] = {}
    for idx, yolo_layer in enumerate(yolo_layers, start=1):
        layer_name = 'YOLO_Layer{}'.format(idx)
        tensorboard_log[layer_name] = {}
        for name, metric in yolo_layer.metrics.items():
            tensorboard_log[layer_name]['{}'.format(name)] = metric
            if idx == 1:
                tensorboard_log['Average_All_Layers']['{}'.format(
                    name)] = metric / len(yolo_layers)
            else:
                tensorboard_log['Average_All_Layers']['{}'.format(
                    name)] += metric / len(yolo_layers)

    return tensorboard_log
