import os
import random

import numpy as np
import pandas as pd
from skimage import io
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class DINODataset(Dataset):
    def __init__(self, csv_path=None, transforms=None, channel=10):
        self.df = pd.read_csv(csv_path)
        self.transform = transforms
        self.channel = channel

    def read_im(self, file_path):
        if file_path.endswith('.npy'):
            image = np.load(file_path, allow_pickle=True)
        else:
            image = io.imread(file_path) ### use other image reading method

        channel_dim = np.argmin(image.shape)
        image = np.moveaxis(image, channel_dim, 0) if channel_dim != 0 else image
        return torch.tensor(image, dtype=torch.float) # to tensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        assert self.df is not None, "Dataloader error: Dataframe (df) can't be None!!!"

        data = self.df.iloc[idx]
        image = self.read_im(data['file_path']) / 255.
        
        C, H, W = image.shape
        image_ = torch.zeros(self.channel, H, W)
        image_[:C] = image[:C]

        image = self.transform(image_)

        c = random.choice(range(C))
        C = [im.shape[0] for im in image]
        # choose random channels
        for i in range(len(image)):
            # c = random.choice(range(C[i]))
            image[i] = image[i][c:c+1]

        return image, C # label is none

