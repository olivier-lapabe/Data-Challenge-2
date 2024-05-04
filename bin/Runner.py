import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from torchvision.models import mobilenet_v3_small
from src.DataLoader.DataLoader import create_dataloaders
from src.utils import define_device


def main():

    #Create Training, Validation and Test datasets
    training_generator, validation_generator, test_generator = create_dataloaders(
        n_val = 20000, 
        batch_size=256, 
        num_workers=0)
    
    # Define device
    device = define_device()
    print("Using device:", device)

    ### Define the model, the loss and the optimizer
    model = torchvision.models.mobilenet_v3_small(num_classes=1)
    model = model.to(device) 
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ### Train the model
    num_epochs = 3
    torch.backends.mps.benchmark = True

    start_time = time.time()

    for i in range(num_epochs):
        start_time_epoch = time.time()
        print(f"Epoch {i+1}")
        epoch_loss = 0

        for batch_idx, (X, y, gender, filename) in tqdm(enumerate(training_generator), total=len(training_generator)):
            X, y = X.to(device), y.to(device) # Transfer to GPU
            y = torch.reshape(y, (len(y), 1))

            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            if torch.isnan(loss).any():
                print(f'NaN detected in loss: filename {filename} - label {y} - y_pred {y_pred}')
                break
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        average_loss = epoch_loss / len(training_generator)
        end_time_epoch = time.time()
        print(f"Epoch {i} - Average loss: {average_loss} - duration: {end_time_epoch - start_time_epoch} seconds")

    end_time = time.time()
    print(f"Average time per epoch: {(end_time - start_time) / num_epochs} seconds")
    
        
if __name__ == "__main__":
    main()