import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("/home/shu/projects/stress-strain-prediction/Curves/Databases/FinalEdited_Cleaned.csv")

feature_cols = ['c','si','mn','p','s','cu','ni','cr','mo','v','nb',
                'ti','al','b','zr','sn','ca','h','fe']
X = df[feature_cols]

ultimate_stress = df.groupby("file")["Sigma_true"].transform("max")
y = df[["Sigma_true","e_true","yield_stress__mpa_"]].copy()
y["ultimate_stress"] = ultimate_stress

# Grouped split
groups = df["file"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Scale inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
mlp = MultiOutputRegressor(
    MLPRegressor(hidden_layer_sizes=(128,64,32),
                 activation='relu',
                 max_iter=500,
                 random_state=42)
)
mlp.fit(X_train, y_train)

# Predict
y_pred = mlp.predict(X_test)

# Evaluate
for i, col in enumerate(y.columns):
    print(f"{col} -> R2: {r2_score(y_test.iloc[:,i], y_pred[:,i]):.3f}, "
          f"MSE: {mean_squared_error(y_test.iloc[:,i], y_pred[:,i]):.3f}")
