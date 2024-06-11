from datetime import datetime
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import torch
from src.utils import metric_fn
from torch.optim.lr_scheduler import StepLR


class Tester:
    # -----------------------------------------------------------------------------
    # __init__
    # -----------------------------------------------------------------------------
    def __init__(self, model, device, loss_fn, optimizer, trainval_dataloaders, test_dataloader, log_directory, scheduler=None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.trainval_dataloaders = trainval_dataloaders
        self.test_dataloader = test_dataloader
        self.log_directory = log_directory
        if scheduler is not None:
            self.scheduler = scheduler

    # -----------------------------------------------------------------------------
    # test
    # -----------------------------------------------------------------------------

    def test(self, num_epochs):
        # TODO: Retrieve best model (hyper-)parameters (incl. num_epochs)
        # TODO: Train on whole Train + Val
        # TODO: Predict on Test
        # TODO: Extract csv for upload
        # TODO: What about shuffling train and val between each epoch
        training_dataloader, validation_dataloader = self.trainval_dataloaders
        combined_dataset = torch.utils.data.ConcatDataset(
            [training_dataloader.dataset, validation_dataloader.dataset])
        combined_dataloader = torch.utils.data.DataLoader(
            combined_dataset, batch_size=training_dataloader.batch_size, shuffle=True)

        self.model.train()                      # Turn into "training mode"
        # Allows PyTorch to automatically select the best algorithm for MPS
        torch.backends.mps.benchmark = True
        # Allows PyTorch to automatically select the best algorithm for CUDA
        torch.backends.cuda.benchmark = True

        # Start training over epochs
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1} / {num_epochs}")

            # Iterate training over mini-batches within epoch
            for X, y, _, filename in combined_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y = torch.reshape(y, (len(y), 1))

                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)

                if torch.isnan(loss).any():
                    print(
                        f'NaN detected in loss: filename {filename} - label {y} - y_pred {y_pred}')
                    break

                # Run gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

        self.predict(self.test_dataloader)

    # -----------------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------------

    def predict(self, test_dataloader):
        self.model.eval()   # Turn into "evaluation mode"

        with torch.no_grad():
            results_list = []  # Keep track of predictions
            # Iterate inference over mini-batches
            for X, _ in test_dataloader:
                X = X.to(self.device)
                y_pred = self.model(X)

                # Keep track of predictions
                for i in range(len(X)):
                    results_list.append({'pred': float(y_pred[i])})

        # Save predictions in csv file
        test_df = pd.DataFrame(results_list)
        test_df.to_csv(
            f"{self.log_directory}/Data_Challenge.csv", header=None, index=None)
