import os
import torch
import pandas as pd
from src.DataLoader.DataLoader import create_test_dataloader
from src.utils import define_device
import torchvision

def main():
    # config
    name_submit = 'train_data_aug1'
    path = 'model_da1.pth'
    result_directory = 'results'
    batch_size = 64
    num_workers = int(os.cpu_count()/2)
    device = define_device()
    normalize = False  # Set to True if the model was trained with normalization
    # if the normalize is not the standar one : mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # change the values in the create_test_dataloader function

    # load the model
    model = torchvision.models.mobilenet_v3_small(num_classes=1)
    model.load_state_dict(torch.load(path))
    model.eval()
    model = model.to(device)


    test_dataloader = create_test_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize)

    with torch.no_grad():
        results_list = []  # Keep track of predictions
        # Iterate inference over mini-batches
        for X, _ in test_dataloader:
            X = X.to(device)
            y_pred = model(X)

            # Keep track of predictions
            for i in range(len(X)):
                results_list.append({'pred': float(y_pred[i])})

    # Save predictions in csv file
    test_df = pd.DataFrame(results_list)
    test_df.to_csv(f"{result_directory}/{name_submit}.csv",
                header=None, index=None)

if __name__ == "__main__":
    main()
