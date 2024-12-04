import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape

### CHARGEMENT DU DATASET ###

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

# Dimensions
input_shape = (target_size[0], target_size[1], 1)  # Images en niveaux de gris
latent_dim = 64  # Taille de l'espace latent

### AUTO ENCODEUR ###

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Conv2DTranspose

input_img = Input(shape=input_shape)

# Encodeur
input_img = Input(shape=input_shape)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)  # Réduction à 64x64

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)  # Réduction à 32x32

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)  # Réduction à 16x16
latent = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

# Décodeur
x = Conv2D(256, (3, 3), activation='relu', padding='same')(latent)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)  # Augmente à 32x32

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)  # Augmente à 64x64

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)  # Augmente à 128x128

output_img = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Fonction de perte
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError

# Charger VGG16 pré-entraîné
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Couche pour la perte perceptuelle
perceptual_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)

def perceptual_loss(y_true, y_pred):
    # Convertir les images en 3 canaux (nécessaire pour VGG16)
    y_true_rgb = tf.image.grayscale_to_rgb(y_true)
    y_pred_rgb = tf.image.grayscale_to_rgb(y_pred)
    
    # Calculer les caractéristiques
    y_true_features = perceptual_model(y_true_rgb)
    y_pred_features = perceptual_model(y_pred_rgb)
    
    # Calculer la MSE entre les caractéristiques
    mse = MeanSquaredError()
    return mse(y_true_features, y_pred_features)


autoencoder = Model(input_img, output_img)

import tensorflow as tf

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

autoencoder.compile(optimizer='adam', loss=perceptual_loss, metrics=[psnr])

autoencoder.summary()

### ENTRAINEMENT ###

# Fractionner les données pour validation
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(noisy_images, clean_images, test_size=0.2, random_state=42)

# Augmentation des données avec des transformations
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Générateur pour augmenter les données
train_generator = datagen.flow(x_train, y_train, batch_size=32)

# Entraînement
history = autoencoder.fit(
    train_generator,
    epochs=20,  # Plus d'époques pour mieux entraîner
    validation_data=(x_val, y_val)
)

### PREDICTIONS ###

# Prédictions sur un échantillon
decoded_imgs = autoencoder.predict(noisy_images[:10])

# Affichage des images bruitées, propres et reconstruites
plt.figure(figsize=(15, 6))
for i in range(10):
    # Image bruitée
    plt.subplot(3, 10, i + 1)
    plt.imshow(noisy_images[i].reshape(target_size), cmap='gray')
    plt.axis('off')
    # Image propre
    plt.subplot(3, 10, i + 11)
    plt.imshow(clean_images[i].reshape(target_size), cmap='gray')
    plt.axis('off')
    # Image reconstruite
    plt.subplot(3, 10, i + 21)
    plt.imshow(decoded_imgs[i].reshape(target_size), cmap='gray')
    plt.axis('off')
plt.show()
