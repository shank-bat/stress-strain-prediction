import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

# Load dataset
df = pd.read_csv("/home/shu/projects/stress-strain-prediction/Curves/Databases/FinalEdited_Cleaned.csv")

# Features (composition only)
feature_cols = ['c','si','mn','p','s','cu','ni','cr','mo','v','nb',
                'ti','al','b','zr','sn','ca','h','fe']
X = df[feature_cols]

# Targets
ultimate_stress = df.groupby("file")["Sigma_true"].transform("max")
y = df[["Sigma_true","e_true","yield_stress__mpa_"]].copy()
y["ultimate_stress"] = ultimate_stress

# Grouped split (by file)
groups = df["file"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Train
rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
for i, col in enumerate(y.columns):
    print(f"{col} -> R2: {r2_score(y_test.iloc[:,i], y_pred[:,i]):.3f}, "
          f"MSE: {mean_squared_error(y_test.iloc[:,i], y_pred[:,i]):.3f}")
