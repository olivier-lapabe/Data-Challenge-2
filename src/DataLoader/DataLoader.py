import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from src.utils import calculate_mean_std


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, image_dir, tranform=transforms.Compose([transforms.ToTensor()]), Test=False):
        """
        Initialize the dataset.
        Args:
            df (pd.DataFrame): Dataframe with the filenames (plus the occlusion and gender labels if not Test).
            image_dir (str): Directory with the images.
            Test (bool, optional): Set to True if the dataset is the test set. Defaults to False.
        """
        self.image_dir = image_dir
        self.df = df
        self.transform = tranform
        self.Test = Test

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.df.loc[index]
        filename = row['filename']
        img = Image.open(f"{self.image_dir}/{filename}")
        # img = cv2.imread(f"{self.image_dir}/{filename}")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X = self.transform(img)

        if not self.Test:
            y = np.float32(row['FaceOcclusion'])
            gender = row['gender']
            return X, y, gender, filename

        else:
            return X, filename


# -----------------------------------------------------------------------------
# create_trainval_dataloaders
# -----------------------------------------------------------------------------
def create_trainval_dataloaders(n_val=20000, batch_size=8, num_workers=0, shuffle_train=True, shuffle_val=False, data_augmentation=False, normalize=False):
    """
    Create training, validation and test dataloaders.

    Args:
        n_val (int, optional): Number of samples in validation set. Defaults to 20000.
        batch_size (int, optional): How many samples per batch to load. Defaults to 8.
        num_workers (int, optional): How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Defaults to 0.
        shuffle_train (bool, optional): Set to True to have the train data reshuffled at every epoch. Defaults to True.
        shuffle_val (bool, optional): Set to True to have the validation data reshuffled at every epoch. Defaults to False.

    Returns:
        training_generator, validation_generator (torch.utils.data.DataLoader): Training and validation dataloaders.
    """
    df_train = pd.read_csv(
        "data/listes_training/data_100K/train_100K.csv", delimiter=' ')
    df_train = df_train.dropna()

    val_transforms = transforms.Compose([transforms.ToTensor()])
    if not data_augmentation:
        train_transforms = val_transforms = transforms.Compose(
            [transforms.ToTensor()])
    elif normalize:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),       # Randomly flip the image horizontally
            # Randomly rotate the image by up to 10 degrees
            transforms.RandomRotation(10),
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])  # Normalize the image
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),       # Randomly flip the image horizontally
            # Randomly rotate the image by up to 10 degrees
            transforms.RandomRotation(10),
            transforms.ToTensor(),  # Convert the image to a tensor
        ])

    df_val = df_train.loc[:n_val].reset_index(drop=True)
    df_train = df_train.loc[n_val:].reset_index(drop=True)

    training_set = Dataset(df_train, "data/crops_100K", train_transforms)
    validation_set = Dataset(df_val, "data/crops_100K", val_transforms)

    training_generator = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    validation_generator = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers)

    return training_generator, validation_generator


# -----------------------------------------------------------------------------
# create_test_dataloader
# -----------------------------------------------------------------------------
def create_test_dataloader(batch_size=8, num_workers=0, shuffle_test=False, normalize=False):
    """
    Create test dataloaders.

    Args:
        batch_size (int, optional): How many samples per batch to load. Defaults to 8.
        num_workers (int, optional): How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Defaults to 0.
        shuffle_test (bool, optional): Set to True to have the test data reshuffled at every epoch. Defaults to False.

    Returns:
        test_generator (torch.utils.data.DataLoader): Test dataloader.
    """
    df_test = pd.read_csv(
        "data/listes_training/data_100K/test_students.csv", delimiter=' ')
    df_test = df_test.dropna()
    if normalize:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    test_set = Dataset(df_test, "data/crops_100K",
                       Test=True, tranform=test_transforms)
    test_generator = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

    return test_generator
