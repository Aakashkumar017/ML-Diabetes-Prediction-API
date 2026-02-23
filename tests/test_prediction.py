import joblib
import pandas as pd

model = joblib.load("models/adaboost_model.pkl")

new_patient = pd.DataFrame(
    [[2,120,70,20,28.5,0.5,35, 28.5*35, 120*28.5]],
    columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "BMI_Age",
        "Glucose_BMI"
    ]
)

prediction = model.predict(new_patient)

print("Prediction:", "Diabetic" if prediction[0]==1 else "Not Diabetic")