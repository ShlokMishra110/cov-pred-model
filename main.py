from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Initialize app
app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace '*' with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your ML model
model = joblib.load("MiniProjectML.pkl")

# Define the expected input data schema
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float  # Add/remove based on your actual model input features

# Prediction endpoint
@app.post("/predict")
def predict(data: ModelInput):
    input_data = [[data.feature1, data.feature2, data.feature3]]
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
