import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape

# Charger les images depuis les dossiers
def load_images(data_dir, target_size=(128, 128)):
    images = []
    for img_name in sorted(os.listdir(data_dir)):  # Assurez-vous que l'ordre est cohérent
        img_path = os.path.join(data_dir, img_name)
        img = Image.open(img_path).convert('L').resize(target_size)  # Convertir en niveaux de gris
        images.append(np.array(img) / 255.0)  # Normaliser entre 0 et 1
    return np.array(images)

# Chemins des dossiers
clean_dir = './dataset/Ground_truth'
noisy_dir = './dataset/Noisy_folder'

# Charger les données
target_size = (128, 128)  # Taille cible pour redimensionner les images
clean_images = load_images(clean_dir, target_size=target_size)
noisy_images = load_images(noisy_dir, target_size=target_size)

# Reshape pour les modèles de deep learning
clean_images = clean_images.reshape(-1, target_size[0], target_size[1], 1)
noisy_images = noisy_images.reshape(-1, target_size[0], target_size[1], 1)

print(f"Nombre d'images propres : {len(clean_images)}, Nombre d'images bruitées : {len(noisy_images)}")
