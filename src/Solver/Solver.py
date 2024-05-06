import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import time
import logging
import os
from datetime import datetime
from src.utils import metric_fn

class Solver:
    # -----------------------------------------------------------------------------
    # __init__
    # -----------------------------------------------------------------------------
    def __init__(self, model, device, loss_fn, optimizer, dataloaders, test_name):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.test_name = test_name
        self.history = {'train_loss': [], 'val_loss': []}
        self.log_directory = None
        self.setup_logger()
        

    # -----------------------------------------------------------------------------
    # setup_logger
    # -----------------------------------------------------------------------------
    def setup_logger(self):
        # Formatting the current time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_directory = f"./results/{current_time}_{self.test_name}"
        
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        logging.basicConfig(filename=f"{self.log_directory}/training_log.log",
                            level=logging.INFO,
                            format='%(asctime)s:%(levelname)s: %(message)s')


    # -----------------------------------------------------------------------------
    # train
    # -----------------------------------------------------------------------------
    def train(self, num_epochs):
        training_dataloader, validation_dataloader = self.dataloaders
        logging.info(f"Test name: {self.test_name}")
        logging.info("--- Dataloader parameters ---")
        logging.info(f"# Validation samples: {len(validation_dataloader.dataset)}")
        logging.info(f"Batch size: {training_dataloader.batch_size}")
        logging.info("--- Training parameters ---")
        logging.info(f"Model: {self.model}")
        logging.info(f"Loss function: {self.loss_fn}")
        logging.info(f"Optimizer: {self.optimizer}")
        logging.info(f"# Epochs: {num_epochs}")
        logging.info(f"Device used: {self.device}")
        logging.info("--- Results ---")

        self.model.train()
        torch.backends.mps.benchmark = True
        torch.backends.cuda.benchmark = True

        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_loss = 0
            logging.info(f"Epoch {epoch+1} / {num_epochs}")

            for X, y, gender, filename in training_dataloader:
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
            logging.info(f"Training - Average loss: {average_loss}")

            self.evaluate(validation_dataloader)

        end_time = time.time()
        logging.info(f"Average training + validation time per epoch: {(end_time - start_time) / num_epochs} seconds")
    

    # -----------------------------------------------------------------------------
    # evaluate
    # -----------------------------------------------------------------------------
    def evaluate(self, dataloader):
        self.model.eval()

        results_list = []
        val_loss = 0
        with torch.no_grad():
            for X, y, gender, filename in dataloader:
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
        logging.info(f"Validation - Average loss: {average_loss}")
        logging.info(f"Validation - Score : {score_val}")
    

    # -----------------------------------------------------------------------------
    # save_model
    # -----------------------------------------------------------------------------
    def save_model(self):
        #TODO: Save model only for epoch where Validation score is minimum
        filename = f"{self.log_directory}/model.pth"
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

    
    # -----------------------------------------------------------------------------
    # test_predict
    # -----------------------------------------------------------------------------
    #def test_predict(self, path):
        #TODO: Load best model
        #TODO: Train on whole Train + Val
        #TODO: Predict on Test
        #TODO: Extract csv for upload
