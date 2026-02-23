# 🩺 Diabetes Prediction API (FastAPI + AdaBoost)

This project deploys a trained **AdaBoost Machine Learning model** using FastAPI to predict whether a patient is diabetic based on medical parameters.

---

## 🚀 System Architecture

```
CLIENT (Streamlit / Postman)
        │
        ▼
FastAPI Endpoint (/predict - POST)
        │
        ▼
Pydantic Validation (Input Schema)
        │
        ▼
Feature Engineering
        │
        ▼
Convert Input → Pandas DataFrame
        │
        ▼
AdaBoost Model (Loaded .pkl)
        │
   ┌────┴───────────────┐
   ▼                    ▼
predict()         predict_proba()
   │                    │
   └──────────┬─────────┘
              ▼
        JSON Response
```

---

## 📌 Workflow Explanation

1. **Client Request**
   User sends patient data via Streamlit UI or Postman.

2. **FastAPI Endpoint (`/predict`)**
   Receives POST request containing patient details.

3. **Data Validation (Pydantic)**
   Ensures correct data types and valid value ranges.

4. **Feature Engineering**

   * `BMI_Age = BMI × Age`
   * `Glucose_BMI = Glucose × BMI`

5. **DataFrame Conversion**
   Input is converted into a Pandas DataFrame (model-compatible format).

6. **Model Prediction**

   * `predict()` → Returns class label (0 or 1)
   * `predict_proba()` → Returns probability score

7. **JSON Response**
   Returns:

   ```json
   {
     "prediction": "Diabetic",
     "diabetes_probability": 78.43
   }
   ```

---

## 🧠 Model Details

* Algorithm: **AdaBoost Classifier**
* Type: Binary Classification
* Target:

  * `0` → Not Diabetic
  * `1` → Diabetic

---

## ▶️ Run the API

```bash
uvicorn main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```