import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet101, ResNet101_Weights
from src.Solver.ROP_Solver import Solver
from src.utils import define_device
from src.DataLoader.HDF5_DataLoader_tensor import create_tensor_dataloaders


def main():
    # Test name (used for the name of the results folder)
    test_name = "resnet101_256_250_ROP_REG"

    # Dataloader parameters
    n_val = 20000
    batch_size = 256
    num_workers = int(os.cpu_count() / 2)

    # Initialize ResNet-101 with pretrained weights
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    # Change to output a single continuous value for regression
    model.fc = nn.Linear(num_ftrs, 1)

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    # Consider using a smaller learning rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 250

    # Create Training, Validation dataloaders
    trainval_dataloaders = create_tensor_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        path="./images_dataset_resnet.hdf5")

    # Define device
    device = define_device()

    # Move model to appropriate device
    model = model.to(device)

    # Train and evaluate the model
    solver = Solver(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        dataloaders=trainval_dataloaders,
        test_name=test_name)
    solver.train(num_epochs=num_epochs)


if __name__ == "__main__":
    main()
