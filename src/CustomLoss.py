import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        # Calculate the weights for each sample
        weights = 1/30 + y_true
        
        # Calculate the squared differences
        squared_diff = (y_pred - y_true) ** 2
        
        # Apply the weights
        weighted_squared_diff = weights * squared_diff
        
        # Compute the mean of the weighted squared differences
        return weighted_squared_diff.mean()