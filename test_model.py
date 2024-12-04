import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

# Répertoires des données
NOISY_FOLDER = "dataset/Noisy_folder"
GROUND_TRUTH = "dataset/Ground_truth"
MODEL_PATH = "autoencoder_denoise.h5"

# Paramètres
IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_images(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0  # Normaliser entre 0 et 1
        images.append(img_array)
    return np.array(images)

def display_results(noisy_images, clean_images, denoised_images, n=10):
    for i in range(n):
        plt.figure(figsize=(12, 4))
        
        # Image bruitée
        plt.subplot(1, 3, 1)
        plt.imshow(noisy_images[i])
        plt.title("Noisy Image")
        plt.axis("off")
        
        # Image prédite (ébruitée)
        plt.subplot(1, 3, 2)
        plt.imshow(denoised_images[i])
        plt.title("Denoised Image")
        plt.axis("off")
        
        # Image propre (vérité terrain)
        plt.subplot(1, 3, 3)
        plt.imshow(clean_images[i])
        plt.title("Ground Truth")
        plt.axis("off")
        
        plt.show()

def calculate_psnr(mse, max_pixel_value=1.0):
    if mse == 0:
        return float('inf')  # Valeur idéale
    return 20 * math.log10(max_pixel_value) - 10 * math.log10(mse)

def main():
    # Charger le modèle avec compile=False
    model = load_model(MODEL_PATH, compile=False)
    print("Modèle chargé depuis", MODEL_PATH)
    
    # Recompiler le modèle avec la même fonction de perte
    model.compile(optimizer='adam', loss='mse')
    
    # Charger les données de test
    noisy_images = load_images(NOISY_FOLDER)
    clean_images = load_images(GROUND_TRUTH)
    _, X_test, _, Y_test = train_test_split(noisy_images, clean_images, test_size=0.2, random_state=42)

    # Prédictions sur les données de test
    denoised_images = model.predict(X_test)

    # Calcul du MSE et PSNR
    mse_values = []
    psnr_values = []

    for i in range(len(Y_test)):
        mse = mean_squared_error(Y_test[i].flatten(), denoised_images[i].flatten())
        psnr = calculate_psnr(mse)
        mse_values.append(mse)
        psnr_values.append(psnr)

    average_mse = np.mean(mse_values)
    average_psnr = np.mean(psnr_values)

    print(f"MSE moyen sur le dataset de test : {average_mse}")
    print(f"PSNR moyen sur le dataset de test : {average_psnr}")

    # Afficher quelques résultats
    display_results(X_test, Y_test, denoised_images)

if __name__ == "__main__":
    main()