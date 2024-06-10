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

# Updated transforms with normalization for ResNet
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_image_tensor(image_path):
    full_path = os.path.join(image_dir, image_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    img = Image.open(full_path).convert('RGB')
    return transform(img)

def batch_process_images(df, batch_size=2500):
    num_batches = (len(df) + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_df = df[i*batch_size:(i+1)*batch_size]
        images = [get_image_tensor(row['filename'])
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
