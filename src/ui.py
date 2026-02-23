import streamlit as st
import requests

st.set_page_config(page_title="Diabetes Prediction System", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to predict diabetes risk")


# -------------------------------
# Input Fields
# -------------------------------
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=30, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict"):

    url = "http://127.0.0.1:8000/predict"

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()

        st.subheader("Prediction Result")
        st.success(f"Prediction: {result['prediction']}")
        st.info(f"Diabetes Probability: {result['diabetes_probability']}%")

        

    else:
        st.error("Error connecting to FastAPI backend")