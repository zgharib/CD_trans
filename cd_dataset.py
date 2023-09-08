# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 01:05:54 2023

@author: zeina
"""

import os
import torch
import numpy as np
from PIL import Image
from skimage import exposure
from torch.utils.data import Dataset



class ChangeDetectionDataset(Dataset):
    def __init__(self, epoch1_folder, epoch2_folder, mask_folder):
        # Filtering only .png files
        self.epoch1_files = sorted([f for f in os.listdir(epoch1_folder) if f.endswith('.png')])
        self.epoch2_files = sorted([f for f in os.listdir(epoch2_folder) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

        self.epoch1_folder = epoch1_folder
        self.epoch2_folder = epoch2_folder
        self.mask_folder = mask_folder

    def __len__(self):
        return len(self.epoch1_files)

    def __getitem__(self, idx):
        # Reading .png files using PIL
        epoch1_img = Image.open(os.path.join(self.epoch1_folder, self.epoch1_files[idx]))
        epoch2_img = Image.open(os.path.join(self.epoch2_folder, self.epoch2_files[idx]))
        mask_img = Image.open(os.path.join(self.mask_folder, self.mask_files[idx]))

        if mask_img.mode == 'RGB':
            mask_img = mask_img.convert('L')  # Convert to grayscale

        # Resizing images using PIL
        epoch1_img = epoch1_img.resize((256, 256))
        epoch2_img = epoch2_img.resize((256, 256))
        mask_img = mask_img.resize((256, 256))

        # Matching the histogram for contrast and brightness
        epoch1_img_array = np.array(epoch1_img)
        epoch2_img_array = np.array(epoch2_img)
        epoch2_img_array = exposure.match_histograms(epoch2_img_array, epoch1_img_array, channel_axis=-1)

        # Transposing to channel-first
        epoch1_img = torch.tensor(epoch1_img_array).float().permute(2, 0, 1) / 255.0
        epoch2_img = torch.tensor(epoch2_img_array).float().permute(2, 0, 1) / 255.0

        # Converting mask to binary
        mask_array = np.array(mask_img)
        mask_array = (mask_array > 0).astype(np.float32)
        mask_img = torch.tensor(mask_array).unsqueeze(0).float()

        sample = {'epoch1': epoch1_img, 'epoch2': epoch2_img, 'mask': mask_img}
        return sample