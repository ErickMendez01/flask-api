from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask_cors import CORS
import logging
from marshmallow import Schema, fields, ValidationError

# Configuración del registro de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializa la aplicación Flask
app = Flask(__name__)

# Configuración de CORS para permitir solicitudes desde el frontend
CORS(app, origins=[""])

# Esquema para validar los datos de entrada
class PredictionSchema(Schema):
    Age = fields.Integer(required=True)
    Marital_Status = fields.String(required=True)
    Education_Level = fields.String(required=True)
    Number_of_Children = fields.Integer(required=True)
    Smoking_Status = fields.String(required=True)
    Physical_Activity_Level = fields.String(required=True)
    Employment_Status = fields.String(required=True)
    Alcohol_Consumption = fields.String(required=True)
    Dietary_Habits = fields.String(required=True)
    Sleep_Patterns = fields.String(required=True)
    History_of_Mental_Illness = fields.String(required=True)
    History_of_Substance_Abuse = fields.String(required=True)
    Family_History_of_Depression = fields.String(required=True)

schema = PredictionSchema()

# Cargar el modelo entrenado desde un archivo
def load_model():
    try:
        logging.info("Cargando el modelo...")
        return joblib.load('random_forest_model.pkl')
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()

# Ruta para manejar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Verificar si el modelo se cargó correctamente
    if model is None:
        logging.error("Modelo no cargado correctamente.")
        return jsonify({'error': 'Modelo no cargado correctamente'}), 500

    try:
        # Obtener y validar los datos de entrada
        data = request.get_json()
        validated_data = schema.load(data)

        # Convertir valores a enteros donde sea necesario
        for key in validated_data:
            try:
                validated_data[key] = int(validated_data[key])
            except ValueError:
                # Si no es un número, se deja como está
                pass

        # Convertir los datos validados a un DataFrame
        input_data = pd.DataFrame([validated_data])

        # Realizar la predicción
        prediction = model.predict(input_data)
        result = "Riesgo de depresión" if prediction[0] == 1 else "No hay riesgo de depresión"

        logging.info("Predicción realizada con éxito.")
        return jsonify({'result': result})

    except ValidationError as err:
        logging.warning(f"Datos inválidos: {err.messages}")
        return jsonify({'error': 'Datos inválidos', 'details': err.messages}), 400

    except Exception as e:
        logging.error(f"Error durante la predicción: {e}")
        return jsonify({'error': 'Hubo un error al hacer la predicción'}), 500

# Ejecutar la aplicación en el puerto 5000
if __name__ == '__main__':
    app.run(debug=True)
