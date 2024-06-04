import os
import torch
import torch.nn as nn
import timm  # Import timm for model architecture
from src.DataLoader.EfficientnetDataLoader import create_trainval_dataloaders
from src.Solver.Solver import Solver
from src.utils import define_device
# If you prefer to use your custom loss, replace nn.MSELoss with CustomLoss
from src.CustomLoss import CustomLoss


def main():
    """
    Run Train_runner.py to:
    - Train a model on the training dataset with chosen (hyper)-parameters.
    - Validate it on the validation dataset to find optimal number of epochs (right between under- and overfitting).
    (Then, run Test_runner.py to predict on the test dataset.)
    """

    # Test name (used for the name of the results folder)
    test_name = "Baseline_EfficientNetB3"

    # Dataloader parameters
    n_val = 20000
    batch_size = 64
    num_workers = int(os.cpu_count() / 2)

    # Training parameters
    # Initialize EfficientNet-B3 for regression (num_classes=1 implies a single regression output)
    model = timm.create_model(
        'efficientnet_b3', pretrained=True, num_classes=1)
    loss_fn = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    # Consider experimenting with the learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 500

    # Create Training, Validation dataloaders
    trainval_dataloaders = create_trainval_dataloaders(
        n_val=n_val,
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
        dataloaders=trainval_dataloaders,
        test_name=test_name)
    solver.train(num_epochs=num_epochs)


if __name__ == "__main__":
    main()
