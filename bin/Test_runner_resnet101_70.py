import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet101, ResNet101_Weights
from src.DataLoader.DataLoader import create_trainval_dataloaders, create_test_dataloader
from src.Tester.Tester import Tester
from src.utils import define_device
from torch.optim.lr_scheduler import CosineAnnealingLR


def main():
    """
    Re-train model with optimized parameters for faster training.
    """
    # Folder for model outputs
    log_directory = "./results/resnet101_70"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # DataLoader parameters
    n_val = 20000
    batch_size = 256  # Larger batch size for faster processing
    num_workers = int(os.cpu_count() / 2)
    data_augmentation = True
    normalize = True

    # Initialize and configure ResNet-101
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    loss_fn = nn.MSELoss()
    # Higher initial learning rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.002, weight_decay=1e-5)

    num_epochs = 70  # Shorter training cycle

    # Cosine Annealing Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)

    # Set up data loaders
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

    # Device configuration and training execution
    device = define_device()
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
