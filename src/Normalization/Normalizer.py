import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Définition des chemins et chargement des données
image_dir = "/Users/guillaume.cazottes/Projets/CEST/CEST-TP/P4/Data-Challenge-2/data/crops_100K"
train_csv = "/Users/guillaume.cazottes/Projets/CEST/CEST-TP/P4/Data-Challenge-2/data/listes_training/data_100K/train_100K.csv"
df = pd.read_csv(train_csv, delimiter=' ')

# Élimination des lignes avec des valeurs manquantes et échantillonnage
df = df.dropna()
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

# Réinitialisation des index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# Transformations d'images
transform = transforms.Compose([transforms.ToTensor()])

def get_image_tensor(image_path):
    full_path = os.path.join(image_dir, image_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    img = Image.open(full_path).convert('RGB')
    return transform(img)

# Fonction pour charger les données en format dictionnaire de listes
def load_data_as_dict(df):
    data_dict = {col: [] for col in df.columns}
    data_dict['image'] = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Loading images"):
        image_tensor = get_image_tensor(row['filename'])
        data_dict['image'].append(image_tensor)
        for col in df.columns:
            data_dict[col].append(row[col])
    return data_dict

# Chargement des données d'entraînement et de validation
train_data_dict = load_data_as_dict(df_train)
val_data_dict = load_data_as_dict(df_val)

# Calcul des moyennes et écarts-types sur les images d'entraînement
train_images = torch.stack(train_data_dict['image'])
train_mean = train_images.mean(dim=(0, 2, 3))
train_std = train_images.std(dim=(0, 2, 3))

# Fonction de normalisation utilisant les paramètres calculés à partir du train
def normalize_images(data_dict, mean, std):
    normalized_images = []
    for img in data_dict['image']:
        normalized_img = (img - mean[:, None, None]) / std[:, None, None]
        normalized_images.append(normalized_img)
    data_dict['image'] = normalized_images
    return data_dict

# Normalisation des ensembles de données
normalized_train_data_dict = normalize_images(train_data_dict, train_mean, train_std)
normalized_val_data_dict = normalize_images(val_data_dict, train_mean, train_std)

# Sauvegarde en fichier .pt
torch.save(normalized_train_data_dict, "normalized_train_data.pt")
torch.save(normalized_val_data_dict, "normalized_val_data.pt")
