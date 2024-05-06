import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import time
from datetime import datetime
from src.utils import metric_fn

class Solver:
    # -----------------------------------------------------------------------------
    # __init__
    # -----------------------------------------------------------------------------
    def __init__(self, model, device, loss_fn, optimizer, dataloaders):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.history = {'train_loss': [], 'val_loss': []}


    # -----------------------------------------------------------------------------
    # train
    # -----------------------------------------------------------------------------
    def train(self, num_epochs):
        self.model.train()
        torch.backends.mps.benchmark = True
        torch.backends.cuda.benchmark = True
        training_dataloader, validation_dataloader = self.dataloaders
        start_time = time.time()

        for epoch in range(num_epochs):
            start_time_epoch = time.time()
            epoch_loss = 0
            print("-----------------------------------------------------------")
            print(f"Epoch {epoch+1} / {num_epochs}")

            for batch_idx, (X, y, gender, filename) in tqdm(enumerate(training_dataloader), total=len(training_dataloader)):
                X, y = X.to(self.device), y.to(self.device)
                y = torch.reshape(y, (len(y), 1))

                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)

                if torch.isnan(loss).any():
                    print(f'NaN detected in loss: filename {filename} - label {y} - y_pred {y_pred}')
                    break

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            average_loss = epoch_loss / len(training_dataloader)
            self.history['train_loss'].append(average_loss)
            end_time_epoch = time.time()
            print(f"Training - Average loss: {average_loss} - Duration: {end_time_epoch - start_time_epoch} seconds")

            self.evaluate(validation_dataloader)

        end_time = time.time()
        print("-----------------------------------------------------------")
        print(f"Average training + validation time per epoch: {(end_time - start_time) / num_epochs} seconds")
    

    # -----------------------------------------------------------------------------
    # evaluate
    # -----------------------------------------------------------------------------
    def evaluate(self, dataloader):
        self.model.eval()
        start_time = time.time()

        results_list = []
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (X, y, gender, filename) in tqdm(enumerate(dataloader), total=len(dataloader)):
                X, y = X.to(self.device), y.to(self.device)
                y = torch.reshape(y, (len(y), 1))

                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)

                if torch.isnan(loss).any():
                    print(f'NaN detected in loss: filename {filename} - label {y} - y_pred {y_pred}')
                    break

                for i in range(len(X)):
                    results_list.append({'pred': float(y_pred[i]),
                                        'target': float(y[i]),
                                        'gender': float(gender[i])
                                        })
                
                val_loss += loss.item()

        results_df = pd.DataFrame(results_list)
        results_male = results_df.loc[results_df["gender"] > 0.5]
        results_female = results_df.loc[results_df["gender"] < 0.5]
        score_val = metric_fn(results_male, results_female)

        average_loss = val_loss / len(dataloader)
        self.history['val_loss'].append(average_loss)
        
        end_time = time.time()
        print(f"Validation - Score : {score_val} - Average loss: {average_loss} - Duration: {end_time - start_time} seconds")
    

    # -----------------------------------------------------------------------------
    # save_model
    # -----------------------------------------------------------------------------
    def save_model(self, test_name, directory='./results'):
        # Format the current time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        full_directory = f"{directory}/{current_time}_{test_name}"

        # Ensure the directory exists
        import os
        if not os.path.exists(full_directory):
            os.makedirs(full_directory)

        # Save the model
        filename = f"{full_directory}/model.pth"
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")
    

    # -----------------------------------------------------------------------------
    # load_model
    # -----------------------------------------------------------------------------
    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")
