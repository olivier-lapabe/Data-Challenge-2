import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from src.DataLoader.DataLoader import create_dataloaders
from src.Solver.Solver import Solver
from src.utils import define_device

def main():

    ## Folder name
    test_name = "Test"

    ## Dataloader parameters
    n_val = 20000
    batch_size=256
    num_workers=32

    ## Training parameters
    model = torchvision.models.mobilenet_v3_large(num_classes=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1

    ## Saving path
    save_path = "./results/"

    #Create Training, Validation and Test datasets
    training_generator, _ = create_dataloaders(
        n_val = n_val, 
        batch_size=batch_size, 
        num_workers=num_workers)
    
    # Define device
    device = define_device()
    print("Using device:", device)
    
    ### Train the model
    model = model.to(device) 
    solver = Solver(model=model, device=device, loss_fn=loss_fn, optimizer=optimizer, dataloader=training_generator)
    solver.train(num_epochs=num_epochs)
    #solver.predict(X)
    #solver.save(test_name=test_name)
        
if __name__ == "__main__":
    main()