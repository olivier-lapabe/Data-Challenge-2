import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class HDF5Dataset(Dataset):
    def __init__(self, file_path, group, transform=None):
        """
        Initialize the dataset from an HDF5 file.
        Args:
            file_path (str): Path to the HDF5 file.
            group (str): Name of the group in the HDF5 file (e.g., 'train', 'val').
            transform (callable, optional): Optional transformation to apply to images.
        """
        self.file_path = file_path
        self.group = group
        self.transform = transform
        self.hdf5_file = h5py.File(self.file_path, 'r')
        self.keys = list(self.hdf5_file[group].keys())

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            image_data = hdf5_file[self.group][self.keys[index]][()]
            image = torch.from_numpy(image_data).float()
            occlusion = np.float32(
                hdf5_file[self.group][self.keys[index]].attrs['FaceOcclusion'])
            gender = hdf5_file[self.group][self.keys[index]].attrs['gender']

            if self.transform:
                image = self.transform(image)

        return image, occlusion, gender, self.keys[index]

    def close(self):
        self.hdf5_file.close()


def create_tensor_dataloaders(batch_size=8, num_workers=0, shuffle_train=True, shuffle_val=False, path="./images_dataset2.hdf5"):
    """
    Create training and validation dataloaders from an HDF5 file.
    Args:
        file_path (str): Path to the HDF5 file.
        batch_size (int): How many samples per batch to load.
        num_workers (int): How many subprocesses to use for data loading.
        shuffle_train (bool): Whether to shuffle the training data.
        shuffle_val (bool): Whether to shuffle the validation data.
    Returns:
        training_generator, validation_generator (torch.utils.data.DataLoader)
    """
    # Transformations for the training dataset
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(
            degrees=10, translate=(0.05, 0.05), scale=None),
        transforms.ToTensor()
    ])

    # No transformation for validation except conversion to tensor
    val_transform = transforms.ToTensor()

    # Initialization of HDF5 datasets with transformations
    training_set = HDF5Dataset(path, 'train', transform=train_transform)
    validation_set = HDF5Dataset(path, 'val', transform=val_transform)

    # Creation of DataLoaders
    training_generator = DataLoader(
        training_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    validation_generator = DataLoader(
        validation_set, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers)

    return training_generator, validation_generator
