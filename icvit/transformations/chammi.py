import os
import torch
from torchvision.transforms import transforms

import numpy as np


def get_mean_std_dataset():
    ## mean, std on training sets
    mean_allen, std_allen = (
        [0.17299628, 0.21203272, 0.06717163],
        [0.31244728, 0.33736905, 0.15192129]
    )
    mean_hpa, std_hpa = (
        [0.08290479, 0.041127298, 0.064044416, 0.08445485],
        [0.16213107, 0.1055938, 0.17713426, 0.1631108],
    )
    mean_cp, std_cp = (
        [0.09957531, 0.19229747, 0.16250895, 0.1824028, 0.14978175],
        [0.1728119, 0.16629605, 0.15171643, 0.14863704, 0.1524553],
    )

    return {
        "CP": (mean_cp, std_cp),
        "Allen": (mean_allen, std_allen),
        "HPA": (mean_hpa, std_hpa),
    }


def get_data_transform(img_size: int):
    """
    if tps_prob > 0, then apply TPS transform with probability tps_prob
    """

    mean_stds = get_mean_std_dataset()
    transform_train = {}
    transform_eval = {}
    for data in ["CP", "Allen", "HPA"]:
        mean_data, std_data = mean_stds[data]
        transform_train_ = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True
                ),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean_data, std_data),
            ]
        )

        transform_eval_ = transforms.Compose(
            [
                transforms.Resize(img_size, antialias=True),
                transforms.CenterCrop(img_size),
                transforms.Normalize(mean_data, std_data),
            ]
        )

        transform_train[data] = transform_train_
        transform_eval[data] = transform_eval_
    
    return transform_train, transform_eval


class CHAMMIAugmentation(object):
    def __init__(
        self,
        is_train: bool,
        img_size: int = 224
    ):
        transform = get_data_transform(img_size)[0]
        self.transform = transform[0] if is_train else transform[1]
        

    def __call__(self, image, chunk="CP"):
        return self.transform[chunk](image)


