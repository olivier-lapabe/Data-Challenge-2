import os
import torch
import torch.nn as nn
import torchvision
from src.DataLoader.DataLoader import create_dataloaders
from src.Solver.Solver import Solver
from src.utils import define_device


def main():

    # Folder name
    test_name = "Test"

    # Dataloader parameters
    n_val = 20000
    batch_size = 64
    num_workers = int(os.cpu_count()/2)

    # Training parameters
    model = torchvision.models.mobilenet_v3_small(num_classes=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 500

    # Create Training, Validation dataloaders
    trainval_generators = create_dataloaders(
        n_val = n_val, 
        batch_size=batch_size, 
        num_workers=num_workers)
    
    # Define device
    device = define_device()
    
    # Train and evaluate the model
    model = model.to(device) 
    solver = Solver(
        model=model, 
        device=device, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        dataloaders=trainval_generators, 
        test_name=test_name)
    solver.train(num_epochs=num_epochs)
        
if __name__ == "__main__":
    main()