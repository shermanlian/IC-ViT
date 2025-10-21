import os
import random

import numpy as np
import pandas as pd
from skimage import io
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from .cell_transformations import CellAugmentation


# source from ContextViT: https://github.com/insitro/ContextViT
def load_meta_data():
    PLATE_TO_ID = {"BR00116991": 0, "BR00116993": 1, "BR00117000": 2}
    FIELD_TO_ID = dict(zip([str(i) for i in range(1, 10)], range(9)))
    WELL_TO_ID = {}
    for i in range(16):
        for j in range(1, 25):
            well_loc = f"{chr(ord('A') + i)}{j:02d}"
            WELL_TO_ID[well_loc] = len(WELL_TO_ID)

    WELL_TO_LBL = {}
    # map the well location to the perturbation label
    # Note that the platemaps are different for different perturbations
    base_path = "~/dataset/wsi/jumpcp/platemap_and_metadata"
    PLATE_MAP = {
        "compound": f"{base_path}/JUMP-Target-1_compound_platemap.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_platemap.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_platemap.tsv",
    }
    META_DATA = {
        "compound": f"{base_path}/JUMP-Target-1_compound_metadata.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_metadata.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_metadata.tsv",
    }
    for perturbation in PLATE_MAP.keys():
        df_platemap = pd.read_parquet(PLATE_MAP[perturbation])
        df_metadata = pd.read_parquet(META_DATA[perturbation])
        df = df_metadata.merge(df_platemap, how="inner", on="broad_sample")

        if perturbation == "compound":
            target_name = "target"
        else:
            target_name = "gene"

        codes, uniques = pd.factorize(df[target_name])
        codes += 1  # set none (neg control) to id 0
        assert min(codes) == 0
        print(f"{target_name} has {len(uniques)} unique values")
        WELL_TO_LBL[perturbation] = dict(zip(df["well_position"], codes))

    return PLATE_TO_ID, FIELD_TO_ID, WELL_TO_ID, WELL_TO_LBL


jumpcp_mean_ = [4.032, 1.566, 3.774, 3.461, 4.172, 6.781, 6.787, 6.778, 0.0, 0.0]
jumpcp_stds_ = [17.318, 12.016, 16.966, 15.065, 17.964, 21.639, 21.671, 21.639, 1.0, 1.0]

jumpcp_mean = [m / 255. for m in jumpcp_mean_]
jumpcp_stds = [s / 255. for s in jumpcp_stds_]

class MultiChannelColorJitter:
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.1, hue=0.5):
        self.color_transform = v2.ColorJitter(brightness=brightness,
                                              contrast=contrast,
                                              saturation=saturation,
                                              hue=hue)

    def __call__(self, image):
        channel_dim = np.argmin(image.shape)
        image = np.moveaxis(image, channel_dim, 1) if channel_dim != 1 else image

        image = torch.tensor(image, dtype=torch.float)
        enhanced_channels = [self.color_transform(c.unsqueeze(0)) for c in image]
        image = torch.cat(enhanced_channels, dim=0).numpy()

        image = np.moveaxis(image, 1, channel_dim) if channel_dim != 1 else image
        return image


# input should be a torch tensor
def train_transform(size):
    print("---------------using train data aug---------------")
    return v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomResizedCrop(size=size, scale=(1.0, 1.0), antialias=True),
        v2.RandomApply(
            [MultiChannelColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
        ),
        v2.ToDtype(torch.float32, scale=False),
        # v2.Normalize(mean=(0.5,), std=(0.5,))
        v2.Normalize(
            (0.370, 0.342, 0.407, 0.025, 0.0002, 0.006, 0.0040, 0.0, 0.0, 0.0), 
            (0.126, 0.124, 0.104, 0.019, 0.0003, 0.004, 0.0017, 1.0, 1.0, 1.0)),
    ])

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        sigma = random.uniform(self.radius_min, self.radius_max)
        ksize = int(3 * sigma + 0.5) * 2 + 1
        return v2.functional.gaussian_blur(img, ksize, sigma)

# input should be a torch tensor
def train_transform(size=224):
    print("---------------using train data aug---------------")
    flip_rotate_list = [v2.RandomHorizontalFlip(),
                        v2.RandomVerticalFlip(),
                        v2.RandomRotation(90),
                        v2.RandomRotation(180),
                        v2.RandomRotation(270)]
    flip_rotate = random.choice(flip_rotate_list)
    return v2.Compose([
        flip_rotate,
        # GaussianBlur(0.5),
        v2.RandomResizedCrop(size=size, scale=(1.0, 1.0), antialias=True),
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize(jumpcp_mean, jumpcp_stds),
    ])


def test_transform(size=224):
    print("---------------using test data aug---------------")
    return v2.Compose([
        v2.Resize(size=size, antialias=True),
        v2.ToDtype(torch.float32, scale=False),
        v2.Normalize(jumpcp_mean, jumpcp_stds),
    ])


class JUMPCPDataset(Dataset):
    def __init__(self, root, cyto_mask_path, split='train', perturbation_type='compound', channel=10):
        super().__init__()
        self.root = root
        self.channel = channel
        df = pd.read_parquet(cyto_mask_path)
        df = self.get_split(df, split)
        self.split = split

        self.data_path = list(df["path"])
        self.data_id = list(df["ID"])
        self.well_loc = list(df["well_loc"])
        self.perturbation_type = perturbation_type
        # self.transform = train_transform() if split == 'train' else test_transform()
        self.transform = CellAugmentation(
                            is_train=(split == 'train'), 
                            global_resize=224,
                            normalization_mean=jumpcp_mean_,
                            normalization_std=jumpcp_stds_)

        self.plate2id, self.field2id, self.well2id, self.well2lbl = load_meta_data()

    def get_split(self, df, split_name, seed=0):
        ###### split copy from ChannelViT #####
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(0.6 * m)
        validate_end = int(0.2 * m) + train_end

        if split_name == "train":
            return df.iloc[perm[:train_end]]
        elif split_name == "valid":
            return df.iloc[perm[train_end:validate_end]]
        elif split_name == "test":
            return df.iloc[perm[validate_end:]]
        else:
            raise ValueError("Unknown split")

    def read_im(self, file_path):
        file_path = file_path.replace('s3://insitro-research-2023-context-vit/jumpcp', self.root)
        image = np.load(file_path, allow_pickle=True) ### jumpcp uses .npy image file
        channel_dim = np.argmin(image.shape)
        image = np.moveaxis(image, channel_dim, -1) if channel_dim != -1 else image
        # return torch.tensor(image, dtype=torch.float) # to tensor
        return image # numpy

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        if self.well_loc[index] not in self.well2lbl[self.perturbation_type]:
            # this well is not labeled
            return None

        image = self.read_im(self.data_path[index]) #/ 255.
        H, W, C = image.shape
        image_ = np.zeros((H, W, self.channel))
        image_[..., :C] = image[..., :C]
        image = self.transform(image_)


        label = self.well2lbl[self.perturbation_type][self.well_loc[index]]

        return {'image': image, 'label': label, 'channels': 8}

    








