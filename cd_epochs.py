# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 01:18:41 2023

@author: zeina
"""

import copy
import torch


def train_model(scheduler, device, model, train_dataloader, val_dataloader, loss_function, optimizer, num_epochs=15):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training Phase
        model.train()
        train_loss = run_epoch(device, train_dataloader, model, loss_function, optimizer, is_training=True)

        # Validation Phase
        model.eval()
        val_loss = run_epoch(device, val_dataloader, model, loss_function, optimizer, is_training=False)

        # Check if this is the best model so far
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        if scheduler:
            scheduler.step()
        torch.cuda.empty_cache()

    print('Training complete.')
    model.load_state_dict(best_model_wts)  # Load the best model weights
    #return model
    torch.cuda.empty_cache()




def run_epoch(device, dataloader, model, loss_function, optimizer, is_training=True):
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