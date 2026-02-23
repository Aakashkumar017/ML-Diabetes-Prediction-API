from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load(
    "models/adaboost_model.pkl"
)

class Details(BaseModel):
    Pregnancies: Annotated[float,Field(ge=0,le=20,description="Enter the Pregnancies",examples=[3,6,9])]
    Glucose: Annotated[float,Field(ge=50,le=300,description="Enter patient Glucose",examples=[50,200,150])]
    BloodPressure: Annotated[float,Field(ge=30,le=200,description="Diastolic blood pressure (mm Hg)",examples=[50,76])]
    SkinThickness: Annotated[float,Field(ge=0,le=100,description="Eter the skinThickness in mm",examples=[20,40,60])]
    BMI: float
    DiabetesPedigreeFunction: float
    Age: Annotated[
        float,
        Field(gt=0, lt=150, description="Enter patient Age", examples=[20, 21, 22])
    ]
@app.get("/")
def home():
    return {"message": "Welcome to the Diabetes Prediction API. Use the /predict endpoint to get predictions."}

@app.post('/predict')
def patient(data: Details):

    bmi_age = data.BMI * data.Age
    glucose_bmi = data.Glucose * data.BMI

    input_df = pd.DataFrame([{
        "Pregnancies": data.Pregnancies,
        "Glucose": data.Glucose,
        "BloodPressure": data.BloodPressure,
        "SkinThickness": data.SkinThickness,
        "BMI": data.BMI,
        "DiabetesPedigreeFunction": data.DiabetesPedigreeFunction,
        "Age": data.Age,
        "BMI_Age": bmi_age,
        "Glucose_BMI": glucose_bmi
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return {
        "prediction": result,
        "diabetes_probability": round(probability * 100, 2)
    }