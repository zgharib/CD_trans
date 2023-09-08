# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 01:11:15 2023

@author: zeina
"""

import torch
from torch import nn
from torchvision.models import resnet18



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