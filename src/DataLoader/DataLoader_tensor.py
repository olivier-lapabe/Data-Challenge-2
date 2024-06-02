import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        Initialize the dataset.
        Args:
            data_dict (dict): Dictionary containing tensors for 'filename', 'FaceOcclusion', 'gender', and 'image'.
        """
        self.filenames = data['filename']
        self.images = data['image']
        self.occlusions = data['FaceOcclusion']
        self.genders = data['gender']

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        image = self.images[index]
        filename = self.filenames[index]
        if self.occlusions is not None and self.genders is not None:
            occlusion = np.float32(self.occlusions[index])
            gender = self.genders[index]
            return image, occlusion, gender, filename
        return image, filename

def create_tensor_dataloaders(batch_size=8, num_workers=0, shuffle_train=True, shuffle_val=False):
    """
    Create training and validation dataloaders from tensor data.
    Args:
        batch_size (int): How many samples per batch to load.
        num_workers (int): How many subprocesses to use for data loading.
        shuffle_train (bool): Whether to shuffle the training data.
        shuffle_val (bool): Whether to shuffle the validation data.
    Returns:
        training_generator, validation_generator (torch.utils.data.DataLoader)
    """
    train_data = torch.load("data/crops_100K_normalized/normalized_train_data.pt")
    val_data = torch.load("data/crops_100K_normalized/normalized_val_data.pt")

    training_set = TensorDataset(train_data)
    validation_set = TensorDataset(val_data)

    training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers)

    return training_generator, validation_generator


