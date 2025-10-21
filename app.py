from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

app = Flask(__name__)

# Chargement du modèle
model_path = "../models/mnist_model.h5"
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
    print("Modèle chargé avec succès")
else:
    print(f"ERREUR : Le modèle n'existe pas à {model_path}")
    model = None

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de vérification de la santé de l'API"""
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    try:
        data = request.json
        
        # Vérification des données
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Conversion et préparation des données
        image_data = np.array(data['image'], dtype=np.float32)
        
        # Vérification de la forme
        if image_data.size != 784:
            return jsonify({'error': 'Image must have 784 values (28x28)'}), 400
        
        image_data = image_data.reshape(1, 784)
        image_data = image_data / 255.0
        
        # Prédiction
        prediction = model.predict(image_data, verbose=0)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': prediction[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    """Retourne les informations du modèle"""
    return jsonify({
        'model_name': 'MNIST Classifier',
        'input_shape': [1, 784],
        'output_classes': 10,
        'model_type': 'Dense Neural Network'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)