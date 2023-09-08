# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 01:39:04 2023

@author: zeina
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix



def evaluate_model(test_dataloader, model, device, threshold=0.4):
    all_preds = []
    all_labels = []

    for batch in test_dataloader:
        epoch1_imgs = batch['epoch1'].to(device)
        epoch2_imgs = batch['epoch2'].to(device)
        masks = batch['mask'].to(device)

        with torch.no_grad():
            outputs, _, _ = model(epoch1_imgs, epoch2_imgs)
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(masks.cpu().numpy().flatten())

    all_preds_bin = [1 if p > threshold else 0 for p in all_preds]
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds_bin).ravel()
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union

    metrics = {
        "Overal Accuracy": accuracy_score(all_labels, all_preds_bin),
        "Precision": precision_score(all_labels, all_preds_bin),
        "Recall": recall_score(all_labels, all_preds_bin),
        "F1-Score": f1_score(all_labels, all_preds_bin),
        "AUC": roc_auc_score(all_labels, all_preds),
        "IoU": iou
        }

    return metrics





def visualize_predictions(epoch1_imgs, epoch2_imgs, true_masks, predicted_masks, num_samples=8):
    fig, axs = plt.subplots(num_samples, 4, figsize=(15, 3*num_samples))

    for i in range(num_samples):
        # Epoch 1 Image
        axs[i, 0].imshow(epoch1_imgs[i].cpu().numpy().transpose((1, 2, 0)))
        axs[i, 0].set_title('Epoch 1 Image')

        # Epoch 2 Image
        axs[i, 1].imshow(epoch2_imgs[i].cpu().numpy().transpose((1, 2, 0)))
        axs[i, 1].set_title('Epoch 2 Image')

        # Ground Truth
        axs[i, 2].imshow(true_masks[i].cpu().numpy().squeeze(), cmap='gray')
        axs[i, 2].set_title('Ground Truth')

        true_mask_np = true_masks[i].cpu().numpy().squeeze()
        predicted_mask_np = predicted_masks[i].cpu().numpy().squeeze()

        # Overlay of the ground truth on the change map
        overlay = np.zeros((*true_mask_np.shape, 3))  # Create an empty RGB image
        overlay[(true_mask_np==1) & (predicted_mask_np!=0)] = [1, 1, 1]  # Right detections as white
        overlay[(true_mask_np==1) & (predicted_mask_np==0)] = [0, 1, 0]  # Undetected pixels as green
        overlay[(true_mask_np==0) & (predicted_mask_np!=0)] = [1, 0, 0]  # Wrong detections as red

        axs[i, 3].imshow(overlay)
        axs[i, 3].set_title('Overlay')

        for ax in axs[i, :]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()