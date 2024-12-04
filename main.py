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

# Dimensions
input_shape = (target_size[0], target_size[1], 1)  # Images en niveaux de gris
latent_dim = 64  # Taille de l'espace latent

# Encodeur
input_img = Input(shape=input_shape)
x = Flatten()(input_img)
x = Dense(512, activation='relu')(x)
x = Dense(latent_dim, activation='relu')(x)

# Decodeur
x = Dense(512, activation='relu')(x)
x = Dense(np.prod(target_size), activation='sigmoid')(x)
output_img = Reshape(target_size + (1,))(x)

# Modèle auto-encodeur
autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

# Fractionner les données pour validation
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(noisy_images, clean_images, test_size=0.2, random_state=42)

# Entraînement
history = autoencoder.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val)
)

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
