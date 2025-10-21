import os
from typing import List, Union
import numpy as np
import torch
import random
import h5py
from omegaconf import DictConfig
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from torchvision import transforms
import socket
import cv2



# tensor([0.4406, 0.4497, 0.4478, 0.4201]) tensor([0.0790, 0.0819, 0.0590, 0.0568])

eurosat_mean = [0.3306, 0.2724, 0.2523, 0.2283, 0.2883, 0.4798, 0.5682, 0.5506, 0.1766, 0.0032, 0.4348, 0.2681, 0.6210]
eurosat_std = [0.0155, 0.0318, 0.0390, 0.0589, 0.0526, 0.0823, 0.1053, 0.1136, 0.0236, 0.0003, 0.0880, 0.0704, 0.1163]

sat6_mean = [0.4406, 0.4497, 0.4478, 0.4201]
sat6_std = [0.0790, 0.0819, 0.0590, 0.0568]

so2sat_mean = np.array([
    -3.5912242e-05,
    -7.658551e-06,
    5.937501e-05,
    2.516598e-05,
    0.044198506,
    0.25761467,
    0.0007556685,
    0.0013503395,
    0.12375654,
    0.109277464,
    0.101086065,
    0.114239536,
    0.15926327,
    0.18147452,
    0.17457514,
    0.1950194,
    0.15428114,
    0.109052904,
]) / 4095

so2sat_std = np.array([
    0.17555329,
    0.17556609,
    0.4599934,
    0.45599362,
    2.855352,
    8.322579,
    2.44937,
    1.464371,
    0.0395863,
    0.047778852,
    0.066362865,
    0.063593246,
    0.07744504,
    0.09099384,
    0.09217117,
    0.10162713,
    0.09989747,
    0.0877891,
]) / 4095


class SatBase(Dataset):
    """So2Sat dataset"""

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    def __init__(
        self,
        path: str,
        transform,
        channels: List[int] = range(18),
        dataset_keys: List[str] = ['eurosat_all', 'sat6', 'so2sat']
    ) -> None:
        """
        Args:
            path: If merged is False, this is the directory containing training.h5/validation.h5.
                  If merged is True, this is the directory where "pretraining.h5" is located.
            transform: The transform to apply.
            channels: List of channel indices.
            split: "train" or other (for validation/test). (Only used when merged is False.)
            merged: When True, use the merged file.
            dataset_key: Only used if merged is False.
        """
        super().__init__()
        self.channels = torch.tensor([c for c in channels])
        self.transform = transform

        # In merged mode, we use the merged file "pretraining.h5" and traverse all keys.
        path = os.path.join(path, "pretraining.h5")
        self.file = h5py.File(path, "r")
        self.path = path
        # Build an index list that contains tuples (key, index) for all keys in the file.
        self.keys = dataset_keys
        self.indices = []
        # self.data = []
        for key in self.keys:
            n = self.file[key].shape[0]
            for i in range(n):
                self.indices.append((key, i))
    
        self.normalize = {
            'eurosat_all': transforms.Normalize(eurosat_mean, eurosat_std),
            'sat6': transforms.Normalize(sat6_mean, sat6_std),
            'so2sat': transforms.Normalize(so2sat_mean, so2sat_std),
        }



    def __getitem__(self, index):
        key, idx = self.indices[index]
        img_chw = self.file[key][idx]
        img_chw = img_chw.astype("float32")
        # Assume image shape is (H, W, C)
        H, W, C = img_chw.shape
        # Reorder dimensions from (H, W, C) to (C, H, W)
        img_chw = np.transpose(img_chw, (2, 0, 1))
        img_chw = torch.tensor(img_chw).float()
        if key == 'sat6':
            img_chw = img_chw / 255.
        else:
            img_chw = img_chw / 4095.

        # Apply transformation.
        img_chw = self.transform(img_chw)
        img_chw = [self.normalize[key](im) for im in img_chw]

        # Perform random channel selection as in the original code.
        c = random.choice(range(C))
        C = [im.shape[0] for im in img_chw]
        for i in range(len(img_chw)):
            img_chw[i] = img_chw[i][c:c+1]

        return img_chw, C

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)
