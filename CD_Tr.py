# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:52:34 2023

@author: zeina
"""

import os
import copy
import numpy as np
from PIL import Image
from skimage import exposure
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Data Preparation

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



# Training Data
epoch1_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/train/A/'
epoch2_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/train/B/'
mask_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/train/label/'

# Instantiating the custom dataset
train_dataset = ChangeDetectionDataset(epoch1_folder, epoch2_folder, mask_folder)# , augmentations=augmentations)

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)

##################
# Validation Data
epoch1_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/test/A/'
epoch2_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/test/B/'
mask_folder = 'C:/Users/zeina/Desktop/My_Paper/Data/new_Levir/test/label/'

val_dataset = ChangeDetectionDataset(epoch1_folder, epoch2_folder, mask_folder)# , augmentations=augmentations)

val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False)

del train_dataset , val_dataset



#------- Model Architecture

class RefinementBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefinementBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear1(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


'''

class ChangeDetectionModel(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_layers=3, num_classes=1):
        super(ChangeDetectionModel, self).__init__()

        self.cnn = resnet18(pretrained=True)
        self.features1 = nn.Sequential(*list(self.cnn.children())[:-2])
        self.features2 = nn.Sequential(*list(self.cnn.children())[:-3])
        self.features3 = nn.Sequential(*list(self.cnn.children())[:-4])
        self.features4 = nn.Sequential(*list(self.cnn.children())[:-5])
        self.features5 = nn.Sequential(*list(self.cnn.children())[:-7])
        self.features6 = nn.Sequential(*list(self.cnn.children())[:-8])

        self.transformer_encoder1 = CustomTransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.transformer_encoder2 = CustomTransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)

        self.upsample1 = nn.ConvTranspose2d(1024, feature_dim, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(768 , feature_dim, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(640 , feature_dim, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(576 , feature_dim, kernel_size=2, stride=2)
        self.upsample5 = nn.ConvTranspose2d(576 , feature_dim, kernel_size=2, stride=2)

        self.output_refine = RefinementBlock(feature_dim, feature_dim)  # Ensure you refine with the correct dimensions
        self.final_conv = nn.Conv2d(feature_dim, 1, kernel_size=1)

        self.activation = nn.Sigmoid()

    def forward(self, epoch1, epoch2):
        #-------Features from RESNET------------------------------------------
        x1_layer = self.features1(epoch1)
        x2_layer = self.features1(epoch2)
        #---------------transformer encoder---------------------------------
        #---------------layer 1 features -----------------------------------
        x1 = x1_layer.view(x1_layer.size(0), x1_layer.size(1), -1).permute(2, 0, 1)
        x2 = x2_layer.view(x2_layer.size(0), x2_layer.size(1), -1).permute(2, 0, 1)

        x1 = self.transformer_encoder1(x1)
        x2 = self.transformer_encoder2(x2)

        x = x1 - x2
        x = x.permute(1, 2, 0).view(x.size(1), 512, 8, 8)
        #-----------------Concatenation-------------------------------------
        xx = x1_layer - x2_layer
        x = torch.cat([xx, x], dim=1)
        #---------------------upsample--------------------------------------
        x = self.upsample1(x)
        #print('x', x.shape)
        #-----------------Concatenation-------------------------------------
        x1_layer = self.features2(epoch1)
        x2_layer = self.features2(epoch2)
        xx = x1_layer - x2_layer
        x = torch.cat([xx, x], dim=1)
        #---------------------upsample&concat-------------------------------
        x = self.upsample2(x)
        x1_layer = self.features3(epoch1)
        x2_layer = self.features3(epoch2)
        xx = x1_layer - x2_layer
        x = torch.cat([xx, x], dim=1)
        #---------------------upsample&concat-------------------------------
        x = self.upsample3(x)
        x1_layer = self.features4(epoch1)
        x2_layer = self.features4(epoch2)
        xx = x1_layer - x2_layer
        x = torch.cat([xx, x], dim=1)
        #---------------------upsample -------------------------------
        x = self.upsample4(x)
        #print('x', x.shape)
        x1_layer = self.features6(epoch1)
        x2_layer = self.features6(epoch2)
        #print('x1_layer', x1_layer.shape)
        xx = x1_layer - x2_layer
        x = torch.cat([xx, x], dim=1)
        #---------------------upsample -------------------------------
        #print('x', x.shape)
        x = self.upsample5(x)
        #x = self.activation(x)
        x = self.output_refine(x)  # Refining the output layer
        x = self.final_conv(x)


        return x


'''
# Or this one:

    
class ChangeDetectionModel(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_layers=3, num_classes=1):
        super(ChangeDetectionModel, self).__init__()

        self.cnn = resnet18(pretrained=True)
        self.features1 = nn.Sequential(*list(self.cnn.children())[:-2])
        self.features2 = nn.Sequential(*list(self.cnn.children())[:-3])
        self.features3 = nn.Sequential(*list(self.cnn.children())[:-4])
        self.features4 = nn.Sequential(*list(self.cnn.children())[:-5])
        self.features5 = nn.Sequential(*list(self.cnn.children())[:-7])

        self.transformer_encoder1 = CustomTransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.transformer_encoder2 = CustomTransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)

        self.upsample1 = nn.ConvTranspose2d(1024, feature_dim, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(768, feature_dim, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(640 , feature_dim, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(576 , feature_dim, kernel_size=2, stride=2)
        self.upsample5 = nn.ConvTranspose2d(576 , num_classes, kernel_size=2, stride=2)

        self.activation = nn.Sigmoid()

    def forward(self, epoch1, epoch2):
        #-------Features from RESNET------------------------------------------
        x1_layer1 = self.features1(epoch1)
        #print('x1_layer1' , x1_layer1.shape )
        x1_layer2 = self.features2(epoch1)
        #print('x1_layer2' , x1_layer2.shape )
        x1_layer3 = self.features3(epoch1)
        #print('x1_layer3' , x1_layer3.shape )
        x1_layer4 = self.features4(epoch1)
        #print('x1_layer4' , x1_layer4.shape )
        x1_layer5 = self.features5(epoch1)
        #print('x1_layer5' , x1_layer5.shape )

        x2_layer1 = self.features1(epoch2)
        x2_layer2 = self.features2(epoch2)
        x2_layer3 = self.features3(epoch2)
        x2_layer4 = self.features4(epoch2)
        x2_layer5 = self.features5(epoch2)

        #---------------diff features:--------------------------------------
        x11 =   x1_layer1 - x2_layer1
        #print('x11' , x11.shape )
        x22 =   x1_layer2 - x2_layer2
        x33 =   x1_layer3 - x2_layer3
        x44 =   x1_layer4 - x2_layer4
        x55 =   x1_layer5 - x2_layer5

        #---------------transformer encoder---------------------------------
        #---------------layer 1 features -----------------------------------
        x1 = x1_layer1.view(x1_layer1.size(0), x1_layer1.size(1), -1).permute(2, 0, 1)
        x2 = x2_layer1.view(x2_layer1.size(0), x2_layer1.size(1), -1).permute(2, 0, 1)

        x1, attn_weights1 = self.transformer_encoder1(x1)
        x2, attn_weights2 = self.transformer_encoder2(x2)

        x = x1 - x2
        x = x.permute(1, 2, 0).view(x.size(1), 512, 8, 8)
        #print("x transformer", x.shape)
        #-----------------Concatenation-------------------------------------
        x = torch.cat([x11, x], dim=1)
        #print("x concat to feature", x.shape)
        #---------------------upsample--------------------------------------
        x = self.upsample1(x)
        #-----------------Concatenation-------------------------------------
        x = torch.cat([x22, x], dim=1)
        #---------------------upsample&concat-------------------------------
        x = self.upsample2(x)
        x = torch.cat([x33, x], dim=1)
        #---------------------upsample&concat-------------------------------
        x = self.upsample3(x)
        x = torch.cat([x44, x], dim=1)
        #print("x up3:", x.shape)
        #---------------------upsample -------------------------------
        x = self.upsample4(x)
        #print("x up4:", x.shape)
        x = torch.cat([x55, x], dim=1)
        #---------------------upsample -------------------------------
        x = self.upsample5(x)
        #print("x up5:", x.shape)


        return x, attn_weights1, attn_weights2


# Train Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ChangeDetectionModel()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"You are using {device}")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001 , weight_decay=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

loss_function = nn.BCEWithLogitsLoss()




def train_model(model, train_dataloader, val_dataloader, loss_function, optimizer, num_epochs=15):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training Phase
        model.train()
        train_loss = run_epoch(train_dataloader, model, loss_function, optimizer, is_training=True)

        # Validation Phase
        model.eval()
        val_loss = run_epoch(val_dataloader, model, loss_function, optimizer, is_training=False)

        # Check if this is the best model so far
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        scheduler.step()
        torch.cuda.empty_cache()

    print('Training complete.')
    model.load_state_dict(best_model_wts)  # Load the best model weights
    #return model
    torch.cuda.empty_cache()




def run_epoch(dataloader, model, loss_function, optimizer, is_training=True):
    running_loss = 0.0

    for batch in dataloader:
        epoch1_imgs = batch['epoch1'].to(device)
        epoch2_imgs = batch['epoch2'].to(device)
        masks = batch['mask'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        with torch.set_grad_enabled(is_training):
            outputs, _, _ = model(epoch1_imgs, epoch2_imgs)
            #outputs = model(epoch1_imgs, epoch2_imgs)
            loss = loss_function(outputs, masks)

            # Backward pass and optimization
            if is_training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * epoch1_imgs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss



train_model(model, train_dataloader, val_dataloader, loss_function, optimizer, num_epochs=10)


