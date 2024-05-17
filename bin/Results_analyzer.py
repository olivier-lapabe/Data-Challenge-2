import os
import torch
import torch.nn as nn
import torchvision
from src.DataLoader.DataLoader import create_trainval_dataloaders
from src.Solver.Solver import Solver
from src.utils import define_device, print_error_decile, print_best_worst_error


def main():
    """
    After training a model with Train_runner.py, run Results_analyzer.py to:
    - Print the mean error by gender decile on the Validation dataset.
    - Show best 5 pictures and worst 5 pictures in terms of error (saved on the root path).
    """

    # Path to the saved model to analyze
    model_path = "./results/2024-05-07_11-39-13_Test/best_model_epoch.pth"

    # Dataloader parameters
    n_val = 20000
    batch_size = 64
    num_workers = int(os.cpu_count()/2)

    # Training parameters
    model = torchvision.models.mobilenet_v3_small(num_classes=1)
    model.load_state_dict(torch.load(model_path))
    loss_fn = nn.MSELoss()

    # Create Training, Validation dataloaders
    _, validation_dataloader = create_trainval_dataloaders(
        n_val = n_val, 
        batch_size=batch_size, 
        num_workers=num_workers)
    
    # Define device
    device = define_device()
    
    # Evaluate the model
    model = model.to(device) 
    solver = Solver(
        model=model, 
        device=device, 
        loss_fn=loss_fn, 
        optimizer=None, 
        dataloaders=None,
        test_name=None)
    _, results_df = solver.evaluate(validation_dataloader=validation_dataloader)

    print_error_decile(results_df)
    print_best_worst_error(results_df)
        
if __name__ == "__main__":
    main()

