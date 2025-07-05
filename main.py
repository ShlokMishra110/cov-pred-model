from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("MiniProjectML.pkl")

app = FastAPI()

# ✅ CORS setup — allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shlokmishra110.github.io"],  # restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# Predict endpoint
@app.post("/predict")
def predict(input: ModelInput):
    try:
        input_data = [[input.feature1, input.feature2, input.feature3]]
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        import traceback
        traceback.print_exc()  # ✅ Show error in Render logs
        return {"error": str(e)}
