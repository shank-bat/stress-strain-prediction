import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

# -----------------------------
# Load and split
# -----------------------------
df = pd.read_csv("/home/shu/projects/stress-strain-prediction/Keypoints/Databases/al_data_cleaned.csv")

X = df.drop(columns=["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)", "class"])
y = df[["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)"]]

categorical = ["Processing"]
numeric = [col for col in X.columns if col not in categorical]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),  # scaling helps XGB a bit
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Model: XGB with multi-output wrapper
# -----------------------------
xgb = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"   # fast, works well for tabular
    )
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb)
])

# -----------------------------
# Train & Evaluate
# -----------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

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
