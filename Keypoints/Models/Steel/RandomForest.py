import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]          # .../Keypoints/Models
DATA_PATH = BASE_DIR.parent / "Databases" / "steel_strength_cleaned.csv"  # go up one (Keypoints/) then into Databases/
df = pd.read_csv(DATA_PATH)

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

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42)))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
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
joblib.dump(model, "steel_rf.pkl")
