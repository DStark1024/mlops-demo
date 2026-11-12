import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MLOps Demo Prediction API")

# Loads the full Pipeline (ColumnTransformer + OHE + Model)
model = joblib.load("models/model.joblib")


class PredictionInput(BaseModel):
    species: str
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput):
    # Convert input to a 1-row DataFrame so ColumnTransformer finds 'species' by name
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]

    # Convert numpy types to standard Python int/str/float for JSON serialization
    if hasattr(prediction, "item"):
        prediction = prediction.item()

    return {"prediction": prediction}
