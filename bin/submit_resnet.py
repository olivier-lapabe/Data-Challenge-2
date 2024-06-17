import os
import torch
import pandas as pd
from torchvision.models import resnet101
from src.DataLoader.RESNET_DataLoader import create_test_dataloader
from src.utils import define_device

# Configurations
name_submit = '2024-06-10_10-19-21_resnet101_256_250_ROP_REG'
path = 'model_best_epoch_on_val.pth'
result_directory = 'results'
batch_size = 64
num_workers = int(os.cpu_count()/2)
device = define_device()

# Load the model architecture
model = resnet101(weights=None)  # Load without pretrained weights
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1)  # Adjust the fully connected layer

# Load your trained model weights
model.load_state_dict(torch.load(path))
model = model.to(device)
model.eval()

# Prepare the test dataloader
test_dataloader = create_test_dataloader(
    batch_size=batch_size,
    num_workers=num_workers
)

# Inference
with torch.no_grad():
    results_list = []  # Keep track of predictions
    for X, _ in test_dataloader:
        X = X.to(device)
        y_pred = model(X)
        # Collect predictions
        for pred in y_pred:
            results_list.append({'pred': float(pred)})

# Save predictions in a CSV file
test_df = pd.DataFrame(results_list)
test_df.to_csv(f"{result_directory}/{name_submit}.csv",
               header=None, index=None)
