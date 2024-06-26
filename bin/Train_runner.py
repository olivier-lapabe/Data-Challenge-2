import os
import torch
import torch.nn as nn
import torchvision
import time
from src.DataLoader.DataLoader import create_trainval_dataloaders
from src.Solver.Solver import Solver
from src.utils import define_device
from src.CustomLoss import CustomLoss
# from src.DataLoader.DataLoader_tensor import create_tensor_dataloaders
from src.DataLoader.HDF5_DataLoader_tensor import create_tensor_dataloaders
from torch.optim.lr_scheduler import StepLR


def main():
    """
    Run Train_runner.py to:
    - Train a model on the training dataset with chosen (hyper)-parameters.
    - Validate it on the validation dataset to find optimal number of epochs (right between under- and overfitting).
    (Then, run Test_runner.py to predict on the test dataset.)
    """

    # Test name (used for the name of the results folder)
    test_name = "resnet101"

    # Dataloader parameters
    n_val = 20000
    batch_size = 256
    num_workers = int(os.cpu_count()/2)
    data_augmentation = True
    normalize = True
    scheduler = None

    # Training parameters
    model = torchvision.models.resnet101(num_classes=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200

    # Learning rate scheduler
    # Uncomment if you want to use a personalised lr scheduler
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    start_time = time.time()

    # Create Training, Validation dataloaders
    trainval_dataloaders = create_trainval_dataloaders(
        n_val=n_val,
        batch_size=batch_size,
        num_workers=num_workers,
        data_augmentation=data_augmentation,
        normalize=normalize)

    # Define device
    device = define_device()

    # Train and evaluate the model
    model = model.to(device)
    solver = Solver(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=trainval_dataloaders,
        test_name=test_name)
    solver.train(num_epochs=num_epochs)


if __name__ == "__main__":
    main()
