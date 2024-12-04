import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Répertoires des données
GROUND_TRUTH_FOLDER = "dataset/Ground_truth"
NOISY_FOLDER = "dataset/Noisy_folder"

# Paramètres
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
EPOCHS = 50

def load_images(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0  # Normaliser entre 0 et 1
        images.append(img_array)
    return np.array(images)

def create_autoencoder(input_shape):
    inputs = Input(shape=input_shape)

    # Encodeur
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)

    # Décodeur
    u3 = UpSampling2D((2, 2))(c4)
    u3 = Concatenate()([u3, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u3)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    u2 = UpSampling2D((2, 2))(c5)
    u2 = Concatenate()([u2, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    u1 = UpSampling2D((2, 2))(c6)
    u1 = Concatenate()([u1, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    return model


def main():
    # Charger les données
    clean_images = load_images(GROUND_TRUTH_FOLDER)
    noisy_images = load_images(NOISY_FOLDER)
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(noisy_images, clean_images, test_size=0.2, random_state=42)
    
    # Créer le modèle
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    autoencoder = create_autoencoder(input_shape)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Entraîner le modèle
    autoencoder.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Sauvegarder le modèle
    autoencoder.save("autoencoder_denoise.h5")
    print("Modèle sauvegardé sous 'autoencoder_denoise.h5'")

if __name__ == "__main__":
    main()
