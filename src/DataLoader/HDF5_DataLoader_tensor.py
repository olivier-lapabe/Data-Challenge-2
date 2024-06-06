import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class HDF5Dataset(Dataset):
    def __init__(self, file_path, group):
        """
        Initialiser le dataset à partir d'un fichier HDF5.
        Args:
            file_path (str): Chemin vers le fichier HDF5.
            group (str): Nom du groupe dans le fichier HDF5 (ex. 'train', 'val').
        """
        self.file_path = file_path
        self.group = group
        self.hdf5_file = h5py.File(self.file_path, 'r')
        self.keys = list(self.hdf5_file[group].keys())

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.keys)

    def __getitem__(self, index):
        try:
            with h5py.File(self.file_path, 'r') as hdf5_file:
                image_data = hdf5_file[self.group][self.keys[index]][()]
                image = torch.from_numpy(image_data).float()
                occlusion = np.float32(
                    hdf5_file[self.group][self.keys[index]].attrs['FaceOcclusion'])
                gender = hdf5_file[self.group][self.keys[index]
                                               ].attrs['gender']
        except KeyError as e:
            print(f"Error accessing data: {e}")
            # Handle the error e.g., by skipping this batch or using a default value
            return None
        return image, occlusion, gender, self.keys[index]

    def close(self):
        self.hdf5_file.close()


def create_tensor_dataloaders(batch_size=8, num_workers=0, shuffle_train=True, shuffle_val=False):
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
    # Initialisation des datasets HDF5
    training_set = HDF5Dataset("./images_dataset2.hdf5", 'train')
    validation_set = HDF5Dataset("./images_dataset2.hdf5", 'val')

    # Création des DataLoaders
    training_generator = DataLoader(
        training_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    validation_generator = DataLoader(
        validation_set, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers)

    return training_generator, validation_generator
