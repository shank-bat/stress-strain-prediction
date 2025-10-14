import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
from time import time
import pathlib

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow import keras
from keras import layers


# -----------------------------
# Paths to datasets (portable)
# -----------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Databases"  # adjust if your CSVs live elsewhere

STEEL_PATH = DATA_DIR / "steel_strength_cleaned.csv"
AL_PATH = DATA_DIR / "al_data_cleaned.csv"

app = FastAPI(title="Stress Prediction Tool API")

# -----------------------------
# Input model
# -----------------------------
class PredictRequest(BaseModel):
    material: str  # "steel" or "aluminium"
    model: str     # "xgb" | "rf" | "nn"
    data: Dict[str, Any]


# -----------------------------
# Utility: get dataset setup
# -----------------------------
def get_data(material):
    if material == "steel":
        df = pd.read_csv(STEEL_PATH)
        # Drop formula column if it exists
        df = df.drop(columns=["formula"], errors="ignore")
        X = df.drop(columns=["yield strength", "tensile strength", "elongation"])
        y = df[["yield strength", "tensile strength", "elongation"]]
        categorical, numeric = [], list(X.columns)

    elif material == "aluminium":
        df = pd.read_csv(AL_PATH)
        X = df.drop(columns=["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)", "class"], errors="ignore")
        y = df[["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)"]]
        categorical, numeric = ["Processing"], [c for c in X.columns if c not in ["Processing"]]
    else:
        raise ValueError("Unknown material")
    return X, y, categorical, numeric


# -----------------------------
# Model builders
# -----------------------------
def build_rf(categorical, numeric):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42)))
    ])
    return model


def build_xgb(categorical, numeric):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    xgb = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            verbosity=0
        )
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", xgb)
    ])
    return model


def build_nn(X, y, categorical, numeric):
    # ensure categorical columns are strings
    if categorical:
        X[categorical] = X[categorical].astype(str)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    Xp = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.2, random_state=42)

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(y.shape[1])
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

    y_pred = model.predict(X_test, verbose=0)
    r2 = r2_score(y_test, y_pred, multioutput="raw_values")
    mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")

    return model, preprocessor, r2, mae


# -----------------------------
# Helpers
# -----------------------------
def build_user_row(data_dict, reference_cols):
    row = {col: data_dict.get(col, 0) for col in reference_cols}
    return pd.DataFrame([row])


# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    material = req.material.lower()
    model_choice = req.model.lower()
    user_data = req.data
    print("\n--- Incoming data from frontend ---")
    print(user_data)
    print("----------------------------------\n")

    start = time()

    try:
        X, y, categorical, numeric = get_data(material)
        ref_cols = list(X.columns)

        # Build user input row
        user_row = build_user_row(user_data, ref_cols)
        if material == "aluminium" and "Processing" not in user_row.columns:
            user_row["Processing"] = "No Processing"

        # ensure categorical data are strings before any preprocessing
        if categorical:
            X[categorical] = X[categorical].astype(str)
            user_row[categorical] = user_row[categorical].astype(str)

        # Select model
        if model_choice == "rf":
            model = build_rf(categorical, numeric)
            model.fit(X, y)
            y_pred_user = model.predict(user_row)
            r2 = mae = [0, 0, 0]  # skip metric calc for brevity

        elif model_choice == "xgb":
            model = build_xgb(categorical, numeric)
            model.fit(X, y)
            y_pred_user = model.predict(user_row)
            r2 = mae = [0, 0, 0]

        elif model_choice == "nn":
            nn_model, preprocessor, r2, mae = build_nn(X, y, categorical, numeric)
            user_row[categorical] = user_row[categorical].astype(str)
            user_input = preprocessor.transform(user_row)
            y_pred_user = nn_model.predict(user_input, verbose=0)
        else:
            raise ValueError("Unknown model")

        # Format prediction output
        if material == "steel":
            result = {
                "Yield Strength": float(y_pred_user[0][0]),
                "Tensile Strength": float(y_pred_user[0][1]),
                "Elongation": float(y_pred_user[0][2]),
            }
        else:
            # Aluminium output order is [Elongation, Tensile, Yield]
            result = {
                "Yield Strength": float(y_pred_user[0][2]),
                "Tensile Strength": float(y_pred_user[0][1]),
                "Elongation": float(y_pred_user[0][0]),
            }

        print(f"\n✅ Training complete | Material: {material} | Model: {model_choice} | Time: {round(time() - start, 2)}s")
        print("Predicted values:", result)

        return {
            "predictions": result,
            "meta": {
                "material": material,
                "model": model_choice,
                "train_time_seconds": round(time() - start, 2),
                "r2": [float(x) for x in r2],
                "mae": [float(x) for x in mae]
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
