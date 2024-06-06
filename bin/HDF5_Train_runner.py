import os
import torch
import torch.nn as nn
import torchvision
from src.DataLoader.DataLoader import create_trainval_dataloaders
from src.Solver.Solver import Solver
from src.utils import define_device
from src.CustomLoss import CustomLoss
# from src.DataLoader.DataLoader_tensor import create_tensor_dataloaders
from src.DataLoader.HDF5_DataLoader_tensor import create_tensor_dataloaders


def main():
    """
    Run Train_runner.py to:
    - Train a model on the training dataset with chosen (hyper)-parameters.
    - Validate it on the validation dataset to find optimal number of epochs (right between under- and overfitting).
    (Then, run Test_runner.py to predict on the test dataset.)
    """

    # Test name (used for the name of the results folder)
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
    # trainval_dataloaders = create_trainval_dataloaders(
    #     n_val = n_val,
    #     batch_size=batch_size,
    #     num_workers=num_workers)

    # TEST
    trainval_dataloaders = create_tensor_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers)
    # FIN DU TEST

    # Define device
    device = define_device()

    # Train and evaluate the model
    model = model.to(device)
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