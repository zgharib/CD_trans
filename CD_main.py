# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:52:34 2023

@author: zeina
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from cd_dataset import ChangeDetectionDataset
from cd_model import ChangeDetectionModel
from cd_epochs import train_model
from cd_test import evaluate_model

# Data Preparation
# ---------------- Training Data

epoch1_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/train/A/'
epoch2_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/train/B/'
mask_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/train/label/'

train_dataset = ChangeDetectionDataset(epoch1_folder, epoch2_folder, mask_folder)# , augmentations=augmentations)
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)

# --------------- Validation Data
epoch1_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/test/A/'
epoch2_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/test/B/'
mask_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/test/label/'

val_dataset = ChangeDetectionDataset(epoch1_folder, epoch2_folder, mask_folder)# , augmentations=augmentations)

val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False)

del train_dataset , val_dataset

#------------------- Model Architecture
# --------   Train Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ChangeDetectionModel()
print(f"You are using {device}")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001 , weight_decay=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

loss_function = nn.BCEWithLogitsLoss()

train_model(scheduler, device, model, train_dataloader, val_dataloader, loss_function, optimizer, num_epochs=10)

#----------------- test and evaluation

test_epoch1_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/test/A/'
test_epoch2_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/test/B/'
test_mask_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/test/label/'

# Instantiating the custom dataset
test_dataset = ChangeDetectionDataset(test_epoch1_folder, test_epoch2_folder, test_mask_folder)

# Creating a DataLoader
test_dataloader = DataLoader(test_dataset, batch_size=9, shuffle=False)
del test_dataset

metrics = evaluate_model(test_dataloader, model, device)

for key, value in metrics.items():
    print(f"{key}: {value}")

