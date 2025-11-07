# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# scaler = joblib.load("scaler.pkl")
# le_gender = joblib.load("label_encoder_gender.pkl")
# le_diabetic = joblib.load("label_encoder_diabetic.pkl")
# le_smoker = joblib.load("label_encoder_smoker.pkl")
# model = joblib.load("best_model.pkl")

# st.set_page_config(page_title = "Insurance Claim Predictor", layout="centered")
# st.title("Health Insurance Payment Prediction App")
# st.write("Enter the details below to estimate your insurance payment amount")

# with st.form("Input_form"):
#     col1, col2 = st.columns(2)
#     with col1:
#         age = st.number_input("Age", min_value=0, max_value=100, value = 30)
#         bmi = st.number_input("BMI", min_value= 10.0, max_value=60.0, value=25.0)
#         children = st.number_input("Number of Children", min_value= 0, max_value= 8, value=0)
#     with col2:
#         bloodpressure = st.number_input("Blood Pressure", min_value= 60, max_value= 200, value= 120)
#         gender = st.selectbox("Gender", options= le_gender.classes_)
#         diabetic = st.selectbox("Diabetic",options= le_diabetic.classes_)
#         smoker = st.selectbox("Smoker", options= le_smoker.classes_)

#     submitted = st.form_submit_button("Predict Payment")

# if submitted:

#     input_data = pd.DataFrame({
#         "age": [age],
#         "gender": [gender],
#         "bmi": [bmi],
#         "bloodpressure": [bloodpressure],
#         "diabetic": [diabetic],
#         "children": [children],
#         "smoker": [smoker]
#     })

#     input_data["gender"] = le_gender.transform(input_data["gender"])
#     input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
#     input_data["smoker"] = le_smoker.transform(input_data["smoker"])

#     num_cols = ["age", "bmi", "bloodpressure", "children"]
#     input_data[num_cols] = scaler.transform(input_data[num_cols])

#     prediction = model.predict(input_data)[0]

#     st.success(f"**Estimated Insurance Payment Amount:** ${prediction:,.2f}")
    
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

# Load models
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")
le_diabetic = joblib.load("label_encoder_diabetic.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
model = joblib.load("best_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Create DataFrame
        input_data = pd.DataFrame({
            "age": [data['age']],
            "gender": [data['gender']],
            "bmi": [data['bmi']],
            "bloodpressure": [data['bloodpressure']],
            "diabetic": [data['diabetic']],
            "children": [data['children']],
            "smoker": [data['smoker']]
        })
        
        # Encode categorical variables
        input_data["gender"] = le_gender.transform(input_data["gender"])
        input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
        input_data["smoker"] = le_smoker.transform(input_data["smoker"])
        
        # Scale numerical features
        num_cols = ["age", "bmi", "bloodpressure", "children"]
        input_data[num_cols] = scaler.transform(input_data[num_cols])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'API is running'})

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)