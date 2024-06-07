import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import h5py
from tqdm import tqdm

# Définition des chemins et chargement des données
image_dir = "./data/crops_100K"
train_csv = "./data/listes_training/data_100K/train_100K.csv"
df = pd.read_csv(train_csv, delimiter=' ')

# Élimination des lignes avec des valeurs manquantes et échantillonnage
df = df.dropna()
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

transform = transforms.Compose([transforms.ToTensor()])


def get_image_tensor(image_path):
    full_path = os.path.join(image_dir, image_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    img = Image.open(full_path).convert('RGB')
    return transform(img)


def calculate_mean_std(df):
    # Créer un DataLoader sans utiliser de batch size, c'est-à-dire charger toutes les données à la fois
    # Cela peut être très gourmand en mémoire, donc assurez-vous que cela est faisable avec votre matériel
    loader = torch.utils.data.DataLoader([get_image_tensor(
        row['filename']) for index, row in df.iterrows()], batch_size=len(df), num_workers=0)

    mean = 0.
    sum_of_squared_error = 0.
    total_samples = 0

    for images in loader:
        batch_samples = images.size(0)  # Nombre total d'images chargées
        images = images.view(batch_samples, images.size(1), -1)

        # Calculer la moyenne sur toutes les images
        mean = images.mean([0, 2])

        # Calculer la variance pour l'écart-type sur toutes les images
        sum_of_squared_error = ((images - mean.unsqueeze(1))**2).sum([0, 2])
        # Nombre total de pixels (H*W par image)
        total_samples = images.size(2) * batch_samples

    # Calcul de l'écart-type global
    std = torch.sqrt(sum_of_squared_error / total_samples)

    return mean, std


mean, std = calculate_mean_std(df_train)


def normalize(tensor):
    return (tensor - mean[:, None, None]) / std[:, None, None]


def batch_process_images(df, batch_size=2500):
    num_batches = (len(df) + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_df = df[i*batch_size:(i+1)*batch_size]
        images = [normalize(get_image_tensor(row['filename']))
                  for _, row in batch_df.iterrows()]
        tensors = torch.stack(images)
        yield tensors, batch_df


# Création du fichier HDF5 pour stocker les données
with h5py.File('images_dataset_normalized.hdf5', 'w') as f:
    train_grp = f.create_group('train')
    val_grp = f.create_group('val')

    def save_batch_to_hdf5(data_tensor, batch_df, group):
        idx_start = group.attrs.get('next_index', 0)
        for i, (tensor, row) in enumerate(zip(data_tensor, batch_df.itertuples(index=False))):
            ds_name = str(idx_start + i)
            ds = group.create_dataset(name=ds_name, data=tensor.numpy())
            ds.attrs['filename'] = row.filename
            ds.attrs['full_path'] = os.path.join(image_dir, row.filename)
            ds.attrs['FaceOcclusion'] = row.FaceOcclusion
            ds.attrs['gender'] = row.gender
        group.attrs['next_index'] = idx_start + len(batch_df)

    for tensors, batch_df in tqdm(batch_process_images(df_train), desc="Saving train images"):
        save_batch_to_hdf5(tensors, batch_df, train_grp)
    for tensors, batch_df in tqdm(batch_process_images(df_val), desc="Saving validation images"):
        save_batch_to_hdf5(tensors, batch_df, val_grp)
    print("Data saved to HDF5 file.")
