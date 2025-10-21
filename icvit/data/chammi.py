import os
from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch
import torchvision
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset
import skimage.io

from channelvit import transformations
from channelvit.data.s3dataset import S3Dataset


class CHAMMI(Dataset):
    """Single cell chunk."""

    ## define all available datasets, used for collate_fn later on

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        split: str,  # train, valid or test
        ssl_flag: bool,
        target_labels: str = "label",
        transform_cfg: DictConfig,
        channels: List[int] = [0, 1, 2, 3, 4],
    ):
        """
        @param csv_path: Path to the csv file with metadata.
        You should copy this file to the dataset folder to avoid modifying other config.

        Note: Allen was renamed to WTC-11 in the paper.
        @param root_dir: root_dir: Directory with all the images.
        @param split: True for training set, False for using all data
        @param target_labels: label column in the csv file
        """

        self.is_train = (split == 'train')
        self.channels = torch.tensor([c for c in channels])

        ## read csv file for the chunk
        self.metadata = pd.read_csv(csv_path)
        ## filter by chunk if chunk is not "morphem70k"
        if is_train:
            self.metadata = self.metadata[self.metadata["train_test_split"] == "Train"]
        else:
            self.metadata = self.metadata[self.metadata["train_test_split"] != "Train"]

        self.metadata = self.metadata.reset_index(drop=True)
        self.root_dir = root_dir
        self.target_labels = target_labels

        ## classes on training set:
        self.train_classes_dict = {
            "BRD-A29260609": 0,
            "BRD-K04185004": 1,
            "BRD-K21680192": 2,
            "DMSO": 3,
            "M0": 4,
            "M1M2": 5,
            "M3": 6,
            "M4M5": 7,
            "M6M7_complete": 8,
            "M6M7_single": 9,
            "golgi apparatus": 10,
            "microtubules": 11,
            "mitochondria": 12,
            "nuclear speckles": 13,
        }

        self.test_classes_dict = None  ## Not defined yet

        self.transform = getattr(transformations, transform_cfg.name)(
            is_train,
            **transform_cfg.args
        )

    @staticmethod
    def _fold_channels(image: np.ndarray, channel_width: int, mode="ignore") -> Tensor:
        """
        Re-arrange channels from tape format to stack tensor
        @param image: (h, w * c)
        @param channel_width:
        @param mode:
        @return: Tensor, shape of  (c, h, w)  in the range [0.0, 1.0]
        """
        # convert to shape of (h, w, c),  (in the range [0, 255])
        output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")

        if mode == "ignore":
            # Keep all channels
            pass
        elif mode == "drop":
            # Drop mask channel (last)
            output = output[:, :, 0:-1]
        elif mode == "apply":
            # Use last channel as a binary mask
            mask = output["image"][:, :, -1:]
            output = output[:, :, 0:-1] * mask
        output = torchvision.transforms.ToTensor()(output)
        return output

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.metadata.loc[idx, "file_path"])
        channel_width = self.metadata.loc[idx, "channel_width"]
        image = skimage.io.imread(img_path)
        image = self._fold_channels(image, channel_width)
        channels = self.channels.numpy()

        if self.is_train:
            label = self.metadata.loc[idx, self.target_labels]
            label = self.train_classes_dict[label]
            label = torch.tensor(label)
        else:
            label = None  ## for now, we don't need labels for evaluation. It will be provided later in evaluation code.

        chunk = self.metadata.loc[idx, "chunk"]
        image = self.transform(chunk, image)

        return (
            image, 
            {
                "channels": channels,
                "chunk": chunk, 
                "label": label
            }
        )


    def __len__(self):
        return len(self.metadata)
