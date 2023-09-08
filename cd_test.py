# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 01:39:04 2023

@author: zeina
"""


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch

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
