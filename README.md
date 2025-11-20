# ML-Powered Health Insurance Cost Estimator

## Overview
This project builds a predictive model that estimates annual health insurance costs for an individual using demographic and lifestyle features (such as age, gender, BMI, smoking status, etc.). It serves as a demonstration of end-to-end data science workflow: data exploration → feature engineering → model training → deployment (flask app) → packaging for inference.

### Key highlights

a. Uses the insurance.csv dataset of actual insurance charges.

b. Pre-processing includes label-encoding categorical features (gender, smoker, region) and scaling numeric features.

c. Model selection and evaluation carried out via a Jupyter Notebook (analysis_model.ipynb).

d. Best performing model serialized (best_model.pkl) along with preprocessing artifacts (scaler, label-encoders) to enable inference.

e. A Flask web application (app.py) allows users to input their details via REST or web form and obtain an insurance cost estimate.

f. Code organisation enables easy enhancement, and the model files permit quick deployment/integration into other applications.

### Technologies & Libraries

Python 3.x

a. pandas, numpy — for data manipulation

b. scikit-learn — for preprocessing, model training & evaluation

c. flask — to build and serve the web application

d. pickle — to serialize model & preprocessing artifacts

e. Jupyter Notebook — for exploratory data analysis and model work

### How to Run / Use the Project

1. #### Clone the repository
   
```
git clone https://github.com/jaid28/ML-Powered-Health-Insurance-Cost-Estimator.git
cd ML-Powered-Health-Insurance-Cost-Estimator
```

2. #### Install dependencies

Create a virtual environment (recommended) and install required packages.
```
pip install -r requirements.txt
```

If a requirements.txt isn’t present yet, you can create one with:
```
pandas numpy scikit-learn flask
```

3. #### Explore the model notebook
   
Launch the Jupyter Notebook analysis_model.ipynb to review the data exploration, model training steps, and evaluation metric

4. #### Run the web application
```
python app.py
```

Navigate to http://127.0.0.1:5000 (or appropriate host/port) in your browser.
Input details (age, gender, BMI, children, smoker status, region, etc.) and submit to get an insurance cost estimate.


### Results & Performance

In the notebook, you will find the model evaluation including metrics such as **RMSE**, **MAE**, and **R²**. The selected best model (serialized in best_model.pkl) is the one that achieved the highest performance, balancing accuracy and generalization (details in the notebook).

### Why This Project Is Valuable

a. Demonstrates data science pipeline from start to finish.

b. Applicable real-world use-case: estimating health insurance costs, which can help individuals plan budgets or assist companies in pricing.

c. Easily extensible: you can substitute different datasets (e.g., country-specific insurance datasets), add features (e.g., habits, genetic risk), or integrate into a full-fledge web app or mobile app.

d. Great for portfolio: shows skills in data preprocessing, feature engineering, model building, deployment.

### Future Enhancements

a. Incorporate more features such as diet, exercise frequency, genetic predisposition, medical history, geolocation.

b. Evaluate additional model types: ensemble models (Random Forest, XGBoost), deep learning architectures.

c. Add hyperparameter tuning (GridSearchCV/RandomizedSearchCV) and cross-validation for more robust modelling.

d. Build an interactive dashboard (e.g., using Streamlit or Dash) for users to visualize cost breakdowns and “what-if” scenarios.

e. Expand deployment: containerize the app (Docker), deploy to cloud (AWS/GCP/Azure) with CI/CD.

f. Add API endpoint for batch predictions and integrate with a mobile app or voice assistant.

g. Add detailed unit-tests and integration tests, logging & monitoring for production readiness.


### Acknowledgements

a. Dataset sourced from the “insurance” dataset commonly used in ML tutorials.

b. Thanks to open-source libraries: pandas, scikit-learn, Flask.

c. Inspired by tutorials on deployment and ML model serving.
