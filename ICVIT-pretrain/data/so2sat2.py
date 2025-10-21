import os
from typing import List, Union
import numpy as np
import torch
import random
import h5py
from omegaconf import DictConfig
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
import socket
import cv2

class So2Sat(Dataset):
    """So2Sat dataset"""

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    def __init__(
        self,
        path: str,
        transform,
        channels: List[int] = range(18),
        split: str = 'train',
        merged: bool = True,
        dataset_key: str = None  # only used when merged==False
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
        self.merged = merged

        if not merged:
            ## read h5py file from path (original behavior)
            if split == "train":
                path = os.path.join(path, "training.h5")
            else:  ## we use the same validation set for both validation and test.
                ## for validation, we use some channels.
                ## for test, we use all channels.
                path = os.path.join(path, "validation.h5")
            self.file = h5py.File(path, "r")
            self.path = path
        else:
            # In merged mode, we use the merged file "pretraining.h5" and traverse all keys.
            path = os.path.join(path, "pretraining.h5")
            self.file = h5py.File(path, "r")
            self.path = path
            # Build an index list that contains tuples (key, index) for all keys in the file.
            self.keys = list(self.file.keys())
            self.indices = []
            for key in self.keys:
                n = self.file[key].shape[0]
                for i in range(n):
                    self.indices.append((key, i))

    def __getitem__(self, index):
        if not self.merged:
            ## Original behavior: concatenate "sen1" and "sen2"
            img_chw = np.concatenate(
                [
                    self.file["sen1"][index].astype("float32"),
                    self.file["sen2"][index].astype("float32"),
                ],
                axis=-1,
            )
            ## reorder the channels to (C, H, W)
            img_chw = np.transpose(img_chw, (2, 0, 1))
            img_chw = torch.tensor(img_chw).float()
        else:
            # In merged mode, retrieve the correct dataset and index.
            key, idx = self.indices[index]
            img_chw = self.file[key][idx]
            img_chw = img_chw.astype("float32")
            # Assume image shape is (H, W, C)
            H, W, C = img_chw.shape
            target_size = 32
            target_channels = 18
            if H == 64 and W == 64:
                # Center crop to 32x32.
                start_h = (H - target_size) // 2
                start_w = (W - target_size) // 2
                img_chw = img_chw[start_h:start_h+target_size, start_w:start_w+target_size, :]
            elif H == 28 and W == 28:
                # Resize to 32x32.
                img_chw = cv2.resize(img_chw, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            elif H == target_size and W == target_size:
                pass
            else:
                raise ValueError(f"Unexpected image size: {H}x{W}")
            # Adjust channels: pad with zeros if fewer than target_channels or truncate if more.
            if C < target_channels:
                pad_width = target_channels - C
                img_chw = np.pad(img_chw, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
            elif C > target_channels:
                img_chw = img_chw[:, :, :target_channels]
            # Reorder dimensions from (H, W, C) to (C, H, W)
            img_chw = np.transpose(img_chw, (2, 0, 1))
            img_chw = torch.tensor(img_chw).float()

        # Apply transformation.
        img_chw = self.transform(img_chw)

        # Perform random channel selection as in the original code.
        # We assume that after transform, img_chw could be a single tensor or a list of tensors.
        if isinstance(img_chw, (list, tuple)):
            # Get channel count from the first element (assuming they are all the same).
            C_val = img_chw[0].shape[0]
            c = random.choice(range(C_val))
            C_list = [im.shape[0] for im in img_chw]  # Keeping original logic.
            for i in range(len(img_chw)):
                img_chw[i] = img_chw[i][c:c+1]
            if not self.merged:
                # In non-merged mode, also process label.
                label = self.file["label"][index].astype(int)
                if sum(label) > 1:
                    raise ValueError("More than one positive")
                for i, y in enumerate(label):
                    if y == 1:
                        label = i
                        break
                return img_chw, C_list
            else:
                return img_chw
        else:
            # If transform returns a single tensor.
            C_val = img_chw.shape[0]
            c = random.choice(range(C_val))
            img_chw = img_chw[c:c+1, :, :]
            if not self.merged:
                label = self.file["label"][index].astype(int)
                # Process label as before.
                if sum(label) > 1:
                    raise ValueError("More than one positive")
                for i, y in enumerate(label):
                    if y == 1:
                        label = i
                        break
                return img_chw, [C_val]
            else:
                return img_chw, [C_val]

    def __len__(self) -> int:
        if not self.merged:
            return len(self.file["label"])
        else:
            return len(self.indices)

    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)
