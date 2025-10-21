# Projet MNIST Deep Learning

## Description
Ce projet couvre l'ensemble du cycle de vie d'un modèle de Deep Learning, de la conception au déploiement. Il utilise le jeu de données MNIST pour la classification des chiffres manuscrits.

## Fonctionnalités
- Entraînement d'un réseau de neurones dense avec Keras/TensorFlow
- Suivi des expérimentations avec MLflow
- API web avec Flask
- Conteneurisation avec Docker
- Pipeline CI/CD avec GitHub Actions

## Structure du projet
```
Project_mnist/
├── src/                 # Code source
│   ├── train_model.py   # Script d'entraînement
│   └── app.py           # API Flask
├── models/              # Modèles sauvegardés
├── notebooks/           # Notebooks Jupyter
├── docker/              # Configuration Docker
├── requirements.txt     # Dépendances
└── README.md            # Ce fichier
```

## Installation

### Prérequis
- Python 3.9 ou supérieur
- Git
- Docker (optionnel pour la conteneurisation)

### Étapes
1. Clonez le repository
```bash
git clone https://github.com/paolo-88-hub/Project_mnist.git
cd Project_mnist
```

2. Créez un environnement virtuel
```bash
python -m venv env
```

3. Activez l'environnement virtuel

**Windows :**
```bash
.\env\Scripts\Activate.ps1
```

**Mac/Linux :**
```bash
source env/bin/activate
```

4. Installez les dépendances
```bash
pip install -r requirements.txt
```

## Utilisation

### Entraîner le modèle
```bash
cd src
python train_model.py
```

### Entraîner avec MLflow
```bash
cd src
python train_model_mlflow.py
```

### Visualiser les expériences MLflow
```bash
mlflow ui
```
Puis ouvrez http://localhost:5000 dans votre navigateur

### Lancer l'API Flask
```bash
cd src
python app.py
```
L'API sera disponible sur http://localhost:5000

### Tester l'API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": [0.1, 0.2, ..., 0.9]}'
```

## Déploiement avec Docker

### Construire l'image
```bash
docker build -f docker/Dockerfile -t mnist-app:latest .
```

### Lancer le conteneur
```bash
docker run -p 5000:5000 mnist-app:latest
```

## Architecture du modèle

Le modèle utilise une architecture de réseau de neurones dense :
- **Couche 1** : Dense (512 neurones, activation ReLU)
- **Dropout** : 0.2
- **Couche 2** : Dense (256 neurones, activation ReLU)
- **Dropout** : 0.2
- **Couche de sortie** : Dense (10 neurones, activation softmax)

**Optimiseur** : Adam  
**Fonction de perte** : Sparse Categorical Crossentropy  
**Métrique** : Accuracy

## Résultats

- **Précision sur le jeu de test** : ~98%
- **Temps d'entraînement** : ~2-5 minutes (10 epochs)

## Technologies utilisées

- **TensorFlow/Keras** : Framework de Deep Learning
- **Flask** : API web
- **MLflow** : Suivi des expérimentations
- **Docker** : Conteneurisation
- **GitHub Actions** : CI/CD
- **NumPy** : Manipulation de données

## Auteur

**[BILOA ABADJECK PAOLO]**  
Étudiant en Génie Informatique, ENSPY  
Département : Génie Informatique  
Date : Septembre 2025

## Licence

Ce projet est réalisé dans le cadre d'un travail pratique académique.

## Contact

Pour toute question, contactez : [paoloabadjeck6@gmail.com]

## Remerciements

- Professeurs : Claude Tinku
- ENSPY - École Nationale Supérieure Polytechnique de Yaoundé