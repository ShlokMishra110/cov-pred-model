from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("MiniProjectML.pkl")

class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float  # Change or add more features if needed

@app.post("/predict")
def predict(data: ModelInput):
    input_data = [[data.feature1, data.feature2, data.feature3]]
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
