import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow import keras
from keras import layers
import pathlib


BASE_DIR = pathlib.Path(__file__).resolve().parents[1]          # .../Keypoints/Models
DATA_PATH = BASE_DIR.parent / "Databases" / "al_data_cleaned.csv"  # go up one (Keypoints/) then into Databases/
                  

df = pd.read_csv(DATA_PATH)

# -----------------------------
# Features (composition + processing)
# -----------------------------
X = df.drop(columns=["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)", "class"])
y = df[["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)"]]

# Define preprocessing
categorical = ["Processing"]
numeric = [col for col in X.columns if col not in categorical]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# -----------------------------
# Build neural network
# -----------------------------
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(3)  # 3 outputs: elongation, tensile, yield
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)

# R² and MAE per output
r2_scores = r2_score(y_test, y_pred, multioutput="raw_values")
mae_scores = mean_absolute_error(y_test, y_pred, multioutput="raw_values")

print("R² scores:", r2_scores)
print("MAE scores:", mae_scores)

# -----------------------------
# Graph output (parity plots)
# -----------------------------
targets = ["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, target in enumerate(targets):
    axes[i].scatter(y_test[target], y_pred[:, i], alpha=0.7, edgecolors="k")
    axes[i].plot([y_test[target].min(), y_test[target].max()],
                 [y_test[target].min(), y_test[target].max()],
                 "r--", lw=2)
    axes[i].set_xlabel(f"Actual {target}")
    axes[i].set_ylabel(f"Predicted {target}")
    axes[i].set_title(f"Parity Plot: {target}")

plt.tight_layout()
plt.show()
