from datetime import datetime
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import torch
from src.utils import error_fn, metric_fn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Solver:
    # -----------------------------------------------------------------------------
    # __init__
    # -----------------------------------------------------------------------------
    def __init__(self, model, device, loss_fn, optimizer, dataloaders, test_name):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10
        )
        self.dataloaders = dataloaders
        self.test_name = test_name
        self.history = {'train_loss': [], 'val_loss': [], 'val_score': []}
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
        logging.info(
            f"# Training samples: {len(training_dataloader.dataset)} - # Validation samples: {len(validation_dataloader.dataset)}")
        logging.info(f"Batch size: {training_dataloader.batch_size}")
        logging.info("--- Training parameters ---")
        logging.info(f"Model: {self.model}")
        logging.info(f"Loss function: {self.loss_fn}")
        logging.info(f"Optimizer: {self.optimizer}")
        logging.info(f"# Epochs: {num_epochs}")
        logging.info(f"Device used: {self.device}")
        logging.info("--- Results ---")

        self.model.train()                      # Turn into "training mode"
        # Allows PyTorch to automatically select the best algorithm for MPS
        torch.backends.mps.benchmark = True
        # Allows PyTorch to automatically select the best algorithm for CUDA
        torch.backends.cuda.benchmark = True

        best_val_score = float('inf')   # Keep track of best validation score
        best_epoch = None               # Keep track of epoch with best validation score

        # Start training over epochs
        start_time = time.time()
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1} / {num_epochs}")
            # Keep track of the cumulated training loss over every mini-batch of the epoch
            epoch_train_loss = 0

            # Iterate training over mini-batches within epoch
            for X, y, _, filename in training_dataloader:
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
                epoch_train_loss += loss.item()

            # Calculate average training loss per batch over the epoch
            average_loss = epoch_train_loss / len(training_dataloader)
            self.history['train_loss'].append(average_loss)
            logging.info(f"Training - Average loss: {average_loss}")

            # Training loop for each epoch
            val_score, val_loss = self.evaluate(validation_dataloader)
            # Update scheduler with validation loss
            self.scheduler.step(val_loss)

            # Loguer le taux d'apprentissage actuel
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Current learning rate: {current_lr}")
            # Keep track of best (minimum) validation score and save corresponding model
            if val_score < best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                torch.save(self.model.state_dict(),
                           f"{self.log_directory}/model_best_epoch_on_val.pth")
                logging.info(
                    f"Model saved - Validation score: {best_val_score}")

        end_time = time.time()
        logging.info("--- Recap training ---")
        logging.info(
            f"Average training + validation time per epoch: {(end_time - start_time) / num_epochs} seconds")
        logging.info(f"Best score: Epoch {best_epoch+1} - {best_val_score}")

        # Plot Training loss, Validation loss and Validation score over epochs
        self.plot_history()

    # -----------------------------------------------------------------------------
    # evaluate
    # -----------------------------------------------------------------------------

    def evaluate(self, validation_dataloader):
        self.model.eval()   # Turn into "evaluation mode"

        results_list = []   # Keep track of predictions for validation score
        val_loss = 0        # Keep track of the cumulated training loss over every mini-batch

        with torch.no_grad():
            # Iterate inference over mini-batches
            for X, y, gender, filename in validation_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y = torch.reshape(y, (len(y), 1))

                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)

                if torch.isnan(loss).any():
                    print(
                        f'NaN detected in loss: filename {filename} - label {y} - y_pred {y_pred}')
                    continue

                # Keep track of the cumulated training loss over every mini-batch
                val_loss += loss.item()

                # Keep track of predictions for validation score
                for i in range(len(X)):
                    results_list.append({'filename': str(filename[i]),
                                        'pred': float(y_pred[i]),
                                         'target': float(y[i]),
                                         'gender': float(gender[i])})

        # Calculate average validation loss per batch
        average_loss = val_loss / len(validation_dataloader)
        self.history['val_loss'].append(average_loss)

        # Calculate validation score
        results_df = pd.DataFrame(results_list)
        results_male = results_df.loc[results_df["gender"] > 0.5]
        results_female = results_df.loc[results_df["gender"] < 0.5]
        val_score = metric_fn(results_male, results_female)
        self.history['val_score'].append(val_score)

        logging.info(f"Validation - Average loss: {average_loss}")
        logging.info(f"Validation - Score : {val_score}")

        return val_score, average_loss

    # -----------------------------------------------------------------------------
    # plot_history
    # -----------------------------------------------------------------------------

    def plot_history(self):
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_loss'], label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.plot(epochs, self.history['val_score'],
                 label='Validation Score', linestyle='--')

        plt.title('Train Loss, Val Loss and Val Score')
        plt.xlabel('Epoch')
        plt.ylabel('Loss/Score')
        plt.legend()

        # Save the plot in the same directory as the logs and model
        plot_filename = f"{self.log_directory}/training_validation_plot.png"
        plt.savefig(plot_filename)
        plt.close()
        logging.info(f"Plot saved to {plot_filename}")
