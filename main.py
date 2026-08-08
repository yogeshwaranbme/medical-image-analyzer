from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# ⚠️ CRITICAL: Allows your Lovable frontend to safely communicate with this Python API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your Lovable website URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the structure of data coming from Lovable
class PatientData(BaseModel):
    age: int
    days_in_hospital: int
    lab_procedures: int
    medications: int

# Load your database and train the ML model exactly as before
conn = sqlite3.connect("hospital_large.db", check_same_thread=False)
df = pd.read_sql_query("SELECT age, days_in_hospital, lab_procedures, medications, readmitted FROM patient_records", conn)
X = df[['age', 'days_in_hospital', 'lab_procedures', 'medications']]
y = df['readmitted']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

@app.post("/predict")
def predict_risk(data: PatientData):
    # Format data for the ML model
    input_vector = [[data.age, data.days_in_hospital, data.lab_procedures, data.medications]]
    
    # Calculate probabilities
    probabilities = model.predict_proba(input_vector)[0]
    readmission_risk = float(probabilities[1]) * 100
    
    # Return JSON response back to Lovable
    return {
        "readmission_risk": round(readmission_risk, 2),
        "risk_tier": "High" if readmission_risk >= 70 else "Medium" if readmission_risk >= 30 else "Low"
    }
