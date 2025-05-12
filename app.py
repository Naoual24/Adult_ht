from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle
try:
    model = joblib.load("adulte.joblib")
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

# Charger les colonnes
try:
    colonnes = joblib.load("colonnes.joblib")
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

        # Vérifier les colonnes manquantes et les ajouter avec des valeurs par défaut
        for col in colonnes:
            if col not in input_data.columns:
                input_data[col] = 0  # Vous pouvez ajuster cela selon le contexte de votre modèle.

        # Réorganiser les colonnes dans le même ordre que lors de l'entraînement
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
