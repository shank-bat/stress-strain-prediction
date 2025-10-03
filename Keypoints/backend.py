# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow import keras  # for NN models

app = FastAPI()

# --------------------------
# Load pre-trained models
# --------------------------
models = {
    "aluminium": {
        "nn": joblib.load("/home/shu/projects/stress-strain-prediction/Keypoints/Models/Aluminium/xgb_alloy_model.pkl"),
        "rf": joblib.load("/home/shu/projects/stress-strain-prediction/Keypoints/Models/Aluminium/aluminium_rf.pkl"),
        "xgb": joblib.load("/home/shu/projects/stress-strain-prediction/Keypoints/Models/Aluminium/aluminium_xgb.pkl"),
    },
    "steel": {
        "nn": keras.models.load_model("/home/shu/projects/stress-strain-prediction/Keypoints/Models/Steel/steel_nn.pkl"),
        "rf": joblib.load("/home/shu/projects/stress-strain-prediction/Keypoints/Models/Steel/steel_rf.pkl"),
        "xgb": joblib.load("/home/shu/projects/stress-strain-prediction/Keypoints/Models/Steel/steel_xgb.pkl"),
    },
}

# --------------------------
# Define input schemas
# --------------------------
class AluminiumInput(BaseModel):
    Processing: str
    Ag: float
    Al: float
    B: float
    Be: float
    Bi: float
    Cd: float
    Co: float
    Cr: float
    Cu: float
    Er: float
    Eu: float
    Fe: float
    Ga: float
    Li: float
    Mg: float
    Mn: float
    Ni: float
    Pb: float
    Sc: float
    Si: float
    Sn: float
    Ti: float
    V: float
    Zn: float
    Zr: float

class SteelInput(BaseModel):
    formula: str
    c: float
    mn: float
    si: float
    cr: float
    ni: float
    mo: float
    v: float
    n: float
    nb: float
    co: float
    w: float
    al: float
    ti: float

# --------------------------
# Prediction route
# --------------------------
@app.post("/predict")
def predict(material: str, model: str, data: dict):
    """
    material: "aluminium" or "steel"
    model: "nn" | "rf" | "xgb"
    data: JSON payload matching the schema
    """
    try:
        chosen_model = models[material][model]

        # Convert dict into numpy array
        features = list(data.values())
        features = np.array(features).reshape(1, -1)

        # Run prediction
        prediction = chosen_model.predict(features)

        return {
            "Yield Strength": float(prediction[0][0]),
            "Tensile Strength": float(prediction[0][1]),
            "Elongation": float(prediction[0][2]),
        }

    except Exception as e:
        return {"error": str(e)}
