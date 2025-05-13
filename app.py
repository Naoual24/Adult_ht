# Importations
from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Chemins des fichiers
drive_path = "/content/drive/MyDrive/ton_dossier/"  # Remplace "ton_dossier" par le nom de ton dossier
model_path = os.path.join(drive_path, "adulte.joblib")
colonnes_path = "colonnes.joblib"  # Le fichier colonnes reste dans le répertoire courant

# Charger le modèle
try:
    model = joblib.load(model_path)
    print("Modèle chargé avec succès depuis Google Drive.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

# Charger les colonnes
try:
    colonnes = joblib.load(colonnes_path)
    print("Colonnes chargées avec succès :", colonnes)
except Exception as e:
    print(f"Erreur lors du chargement des colonnes : {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        data = {
            'age': int(request.form['age']),
            'workclass': request.form['workclass'],
            'education': request.form['education'],
            'marital.status': request.form['marital_status'],
            'occupation': request.form['occupation'],
            'hours.per.week': int(request.form['hours_per_week']),
            'capital.gain': int(request.form['capital_gain']),
            'capital.loss': int(request.form['capital_loss']),
            'sex': request.form['sex']
        }

        # Créer le DataFrame d'entrée
        input_data = pd.DataFrame([data])

        # Ajouter les colonnes manquantes
        for col in colonnes:
            if col not in input_data.columns:
                input_data[col] = 0

        # Réorganiser les colonnes
        input_data = input_data[colonnes]

        # Prédiction
        prediction = model.predict(input_data)[0]
        result = "Revenu > 50K$" if prediction == 1 else "Revenu ≤ 50K$"
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        result = f"Erreur lors de la prédiction : {str(e)}"
        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
