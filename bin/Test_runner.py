import os
import torch
import torch.nn as nn
import torchvision
from src.DataLoader.DataLoader import create_trainval_dataloaders, create_test_dataloader
from src.Tester.Tester import Tester
from src.utils import define_device
from src.CustomLoss import CustomLoss


def main():
    """
    When optimized model has been found with Train_runner.py, use Test_runner.py to:
    - Re-train model with optimized (hyper-)parameters, but this time on training + validation set combined.
    - Predict on test set and create csv output.
    """
    # Folder name where to find the model to be retrained
    log_directory = "./results/2024-05-07_11-39-13_Test"

    # Dataloader parameters
    n_val = 20000
    batch_size = 64
    num_workers = int(os.cpu_count()/2)

    # Training parameters to be copied from the model to be retrained
    model = torchvision.models.mobilenet_v3_small(num_classes=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 403

    # Create Training, Validation dataloaders
    trainval_dataloaders = create_trainval_dataloaders(
        n_val = n_val, 
        batch_size=batch_size, 
        num_workers=num_workers)
    test_dataloader = create_test_dataloader(
        batch_size=batch_size, 
        num_workers=num_workers)
    
    # Define device
    device = define_device()
    
    # Train the model with train+val, predict and create csv output
    model = model.to(device) 
    tester = Tester(
        model=model, 
        device=device, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        trainval_dataloaders=trainval_dataloaders, 
        test_dataloader=test_dataloader,
        log_directory=log_directory)
    tester.test(num_epochs=num_epochs)
        
if __name__ == "__main__":
    main()