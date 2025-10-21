import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.tensorflow

# Configuration des paramètres
EPOCHS = 10
BATCH_SIZE = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

# Chargement du jeu de données MNIST
print("Chargement du jeu de données MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation des données
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Redimensionnement
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Démarrage de la session de suivi MLflow
with mlflow.start_run():
    # Enregistrement des paramètres
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    
    print("Construction et entraînement du modèle...")
    
    # Construction du modèle
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compilation
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entraînement
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    
    # Évaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Enregistrement des métriques
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("train_loss", history.history['loss'][-1])
    
    print(f"\nPrécision sur les données de test : {test_acc:.4f}")
    
    # Sauvegarde du modèle avec MLflow
    mlflow.keras.log_model(model, "mnist-model")
    
    print("Modèle enregistré dans MLflow")