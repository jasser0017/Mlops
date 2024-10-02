import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import joblib
import pandas as pd
from typing import Optional



# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables or set defaults
MODEL_PATH = os.getenv('MODEL_PATH', 'models/logistic_regression_model.joblib')

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0.0",
    description="An API to predict survival on the Titanic based on passenger data."
)

# Load the saved pipeline
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}.")
    raise
except Exception as e:
    logger.error(f"Error loading the model: {e}")
    raise


class PassengerFeatures(BaseModel):
    Pclass: int = Field(..., gt=0, description="Passenger class (1, 2, or 3)")
    Sex: str = Field(..., description="Sex (male or female)")
    Age: float = Field(..., gt=0, description="Age of the passenger")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., gt=0, description="Ticket fare")
    Embarked: str = Field(..., description="Port of embarkation (C, Q, or S)")

@app.get("/", summary="Root Endpoint", description="Welcome to the Titanic Survival Prediction API.")
def read_root():
    return {"message": "Welcome to the Titanic Survival Prediction API. Use /predict to get predictions."}

@app.post("/predict", summary="Predict Titanic Survival", description="Predict the survival of a Titanic passenger.")
async def predict_survival(features: PassengerFeatures):
    try:
        input_data = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
        logger.info(f"Received input data: {input_data.to_dict(orient='records')}")
        
        prediction = model.predict(input_data)[0]
        logger.info(f"Prediction: {prediction}")
        
        return {"prediction": "Survived" if prediction == 1 else "Not Survived"}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health", summary="Health Check", description="Check if the API is running.")
def health_check():
    return {"status": "API is running"}
