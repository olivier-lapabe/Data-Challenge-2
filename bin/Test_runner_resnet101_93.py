import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet101, ResNet101_Weights
from src.DataLoader.DataLoader import create_trainval_dataloaders, create_test_dataloader
from src.Tester.Tester import Tester
from src.utils import define_device
from torch.optim.lr_scheduler import StepLR


def main():
    """
    When optimized model has been found with Train_runner.py, use Test_runner.py to:
    - Re-train model with optimized (hyper-)parameters, but this time on training + validation set combined.
    - Predict on test set and create csv output.
    """
    # Folder name where to find the model to be retrained
    log_directory = "./results/resnet101_93"

    # if the folder does not exist, create it
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Dataloader parameters
    n_val = 20000
    batch_size = 128
    num_workers = int(os.cpu_count()/2)
    data_augmentation = True
    normalize = True

    # Initialize ResNet-101 with pretrained weights
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    # Adjust to output a single continuous value for regression
    model.fc = nn.Linear(num_ftrs, 1)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=1e-5)
    num_epochs = 93  # Adjusted number of epochs

    # Learning rate scheduler
    # Decays the learning rate by a factor of 0.1 every 31 epochs
    scheduler = StepLR(optimizer, step_size=31, gamma=0.1)

    # Create Training, Validation dataloaders
    trainval_dataloaders = create_trainval_dataloaders(
        n_val=n_val,
        batch_size=batch_size,
        num_workers=num_workers,
        data_augmentation=data_augmentation,
        normalize=normalize)
    test_dataloader = create_test_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize)

    # Define device
    device = define_device()

    # Train the model with train+val, predict and create csv output
    model = model.to(device)
    tester = Tester(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        trainval_dataloaders=trainval_dataloaders,
        test_dataloader=test_dataloader,
        log_directory=log_directory)
    tester.test(num_epochs=num_epochs)


if __name__ == "__main__":
    main()
