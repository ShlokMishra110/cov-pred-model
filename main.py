from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("MiniProjectML.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

# Allow frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define expected input format (must match frontend form names)
class ModelInput(BaseModel):
    age: float
    ddimer: float
    crp: float
    ferritin: float
    oxygen: float

@app.post("/predict")
def predict(input: ModelInput):
    try:
        # Convert input into a 2D list for model
        input_data = [[
            input.age,
            input.ddimer,
            input.crp,
            input.ferritin,
            input.oxygen
        ]]
        
        # Apply scaling before prediction
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
