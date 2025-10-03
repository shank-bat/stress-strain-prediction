import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("/home/shu/projects/stress-strain-prediction/Curves/Databases/FinalEdited_Cleaned.csv")

feature_cols = ['c','si','mn','p','s','cu','ni','cr','mo','v','nb',
                'ti','al','b','zr','sn','ca','h','fe']
X = df[feature_cols]

ultimate_stress = df.groupby("file")["Sigma_true"].transform("max")
y = df[["Sigma_true","e_true","yield_stress__mpa_"]].copy()
y["ultimate_stress"] = ultimate_stress

data = X.join(y).dropna()
X = data[feature_cols]
y = data[["Sigma_true","e_true","yield_stress__mpa_","ultimate_stress"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multi-output with XGB
xgb = MultiOutputRegressor(
    XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, tree_method="hist")
)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print("R2 scores:", r2_score(y_test, y_pred, multioutput='raw_values'))
print("MSE:", mean_squared_error(y_test, y_pred, multioutput='raw_values'))
