import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow import keras
from keras import layers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Load data
df = pd.read_csv("/home/shu/projects/stress-strain-prediction/Keypoints/Databases/steel_strength_cleaned.csv")

X = df.drop(columns=["yield strength", "tensile strength", "elongation"])
y = df[["yield strength", "tensile strength", "elongation"]]

categorical = ["formula"]
numeric = [col for col in X.columns if col not in categorical]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Build neural network
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(3)  # yield, tensile, elongation
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=1)

y_pred = model.predict(X_test)

print("R²:", r2_score(y_test, y_pred, multioutput="raw_values"))
print("MAE:", mean_absolute_error(y_test, y_pred, multioutput="raw_values"))

# Plot parity plots
targets = ["yield strength", "tensile strength", "elongation"]
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
import joblib
joblib.dump(model, "steel_nn.pkl")
