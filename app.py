from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load models once at startup
try:
    scaler = joblib.load("scaler.pkl")
    le_gender = joblib.load("label_encoder_gender.pkl")
    le_diabetic = joblib.load("label_encoder_diabetic.pkl")
    le_smoker = joblib.load("label_encoder_smoker.pkl")
    model = joblib.load("best_model.pkl")
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)  # force=True handles missing Content-Type
        
        input_data = pd.DataFrame([{
            "age": data['age'],
            "gender": data['gender'],
            "bmi": data['bmi'],
            "bloodpressure": data['bloodpressure'],
            "diabetic": data['diabetic'],
            "children": data['children'],
            "smoker": data['smoker']
        }])

        # Encoding
        input_data["gender"] = le_gender.transform(input_data["gender"])
        input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
        input_data["smoker"] = le_smoker.transform(input_data["smoker"])

        # Scaling
        num_cols = ["age", "bmi", "bloodpressure", "children"]
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        prediction = model.predict(input_data)[0]

        return jsonify({
            'prediction': round(float(prediction), 2),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health')
def health():
    return jsonify({'status': 'API is running smoothly!'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
