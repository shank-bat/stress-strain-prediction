import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
# Load cleaned dataset
df = pd.read_csv("/home/shu/projects/stress-strain-prediction/Keypoints/Databases/al_data_cleaned.csv")

# Features and targets
X = df.drop(columns=["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)", "class"])
y = df[["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)"]]

# Preprocess categorical column
categorical = ["Processing"]
numeric = [col for col in X.columns if col not in categorical]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model (Random Forest wrapped for multi-output regression)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42)))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R² score:", r2_score(y_test, y_pred, multioutput="raw_values"))
print("MAE:", mean_absolute_error(y_test, y_pred, multioutput="raw_values"))
import joblib
joblib.dump(model, "aluminium_rf.pkl")
