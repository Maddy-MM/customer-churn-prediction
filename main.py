from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model('churn_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = FastAPI()

class CustomerData(BaseModel):
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float
    Geography_Germany: float
    Geography_Spain: float
    Gender_Male: float

@app.get("/")
def home():
    return {
        "message": "Credit Card Customer Churn Prediction API",
        "endpoints": {
            "predict": "/predict",
            "about": "/about"
        }
    }

@app.get("/about")
def info():
    return {
        "project": "Credit Card Customer Churn Prediction API",
        "description": "This API predicts whether a credit card customer is likely to churn using a trained Artificial Neural Network (ANN) model.",
        "model_type": "Artificial Neural Network (TensorFlow/Keras)",
        "input_features": [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Geography_Germany",
            "Geography_Spain",
            "Gender_Male"
        ],
        "output": {
            "churn_probability": "Probability between 0 and 1",
            "churn_prediction": "0 = No Churn, 1 = Churn"
        },
        "author": "Madhav",
        "version": "1.0"
    }

@app.post("/predict/")
async def predict_churn(data: CustomerData):

    input_df = pd.DataFrame([data.model_dump()])

    scaled_input = scaler.transform(input_df)

    prediction_proba = model.predict(scaled_input)[0][0]
    prediction_class = (prediction_proba >= 0.5).astype(int)

    return {
        "churn_probability": float(prediction_proba),
        "churn_prediction": int(prediction_class)
    }