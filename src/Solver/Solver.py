import torch
import torch.nn as nn
from tqdm import tqdm
import time
from datetime import datetime

class Solver:
    def __init__(self, model, device, loss_fn, optimizer, dataloader):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.history = {'train_loss': []}

    def train(self, num_epochs):
        self.model.train()
        torch.backends.mps.benchmark = True
        torch.backends.cuda.benchmark = True
        start_time = time.time()

        for epoch in range(num_epochs):
            start_time_epoch = time.time()
            epoch_loss = 0
            print(f"Epoch {epoch+1} / {num_epochs}")

            for batch_idx, (X, y, gender, filename) in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
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

            average_loss = epoch_loss / len(self.dataloader)
            self.history['train_loss'].append(average_loss)
            end_time_epoch = time.time()
            print(f"Epoch {epoch} - Average loss: {average_loss} - Duration: {end_time_epoch - start_time_epoch} seconds")

        end_time = time.time()
        print(f"Average time per epoch: {(end_time - start_time) / num_epochs} seconds")
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            return self.model(X)
    
    def save_model(self, test_name, directory='./results'):
        # Format the current time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{directory}/{current_time}_{test_name}/model.pth"
        
        # Ensure the directory exists
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the model
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
