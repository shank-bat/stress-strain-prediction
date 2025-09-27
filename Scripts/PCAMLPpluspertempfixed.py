#!/usr/bin/env python3
"""
Per-temperature training with adaptive PCA:
- n_components shrinks based on specimen count per temp.
"""

import os, re, random, joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# -----------------------------
# Paths & params
# -----------------------------
CSV_PATH = "/home/shu/projects/stress-strain-prediction/OutputData/aluminium_6061_curves.csv"
OUT_DIR = "/home/shu/projects/stress-strain-prediction/models_per_temp"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
REQUESTED_PCA = 20
BATCH_SIZE = 16
EPOCHS = 400
LR = 2e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# -----------------------------
# Helpers
# -----------------------------
def try_make_ohe(**kwargs):
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)

class PCARegDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Load & pivot data
# -----------------------------
df = pd.read_csv(CSV_PATH)
pivot = df.pivot_table(
    index=["specimen_type", "temperature", "lot", "specimen_id", "file_name"],
    columns="strain", values="stress"
).reset_index()

# Identify strain cols
strain_cols = [c for c in pivot.columns if re.match(r"^\d+(\.\d+)?$", str(c))]
strain_cols = sorted(strain_cols, key=lambda x: float(x))
strain_grid = np.array([float(c) for c in strain_cols])

# -----------------------------
# Train per temperature
# -----------------------------
temps = sorted(pivot["temperature"].unique())
results = []

for temp in temps:
    subset = pivot[pivot["temperature"] == temp].copy()
    n_samples = subset.shape[0]
    print(f"\n=== Training model for Temperature {temp} ({n_samples} specimens) ===")
    
    # Target curves
    Y = subset[strain_cols].values.astype(np.float32)
    global_mean = np.nanmean(Y, axis=0)
    for i in range(Y.shape[0]):
        if np.any(np.isnan(Y[i])):
            valid = ~np.isnan(Y[i])
            if valid.sum() >= 2:
                Y[i] = np.interp(strain_grid, strain_grid[valid], Y[i,valid])
            else:
                Y[i] = np.where(np.isnan(Y[i]), global_mean, Y[i])
    
    y_scaler = StandardScaler()
    Y_scaled = y_scaler.fit_transform(Y)
    
    # Adaptive PCA size
    n_pca = min(REQUESTED_PCA, n_samples // 2, Y.shape[1])
    if n_pca < 2:
        n_pca = 2
    print(f"  Using {n_pca} PCA components")
    
    pca = PCA(n_components=n_pca)
    Y_pca = pca.fit_transform(Y_scaled)
    
    # Features (only specimen_type + lot matter now)
    X_meta = subset[["specimen_type","lot"]].astype(str)
    ohe = try_make_ohe(handle_unknown="ignore")
    X = ohe.fit_transform(X_meta)
    
    # Train/val split
    if n_samples > 5:
        X_train, X_val, y_train, y_val, Y_train_orig, Y_val_orig = train_test_split(
            X, Y_pca, Y, test_size=0.2, random_state=SEED
        )
    else:
        X_train, X_val, y_train, y_val, Y_train_orig, Y_val_orig = X, X, Y_pca, Y_pca, Y, Y
    
    train_ds = PCARegDataset(X_train, y_train)
    val_ds   = PCARegDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MLP(X.shape[1], Y_pca.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    
    best_val = float("inf")
    for epoch in range(1,EPOCHS+1):
        model.train()
        for xb,yb in train_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred,yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred,yb).item()*xb.size(0)
        val_loss /= len(val_ds)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR,f"model_temp{temp}.pt"))
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}, Val Loss (PCA space): {val_loss:.4f}")
    
    # Evaluate R²
    model.load_state_dict(torch.load(os.path.join(OUT_DIR,f"model_temp{temp}.pt")))
    model.eval()
    with torch.no_grad():
        preds_pca = model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    pred_scaled = pca.inverse_transform(preds_pca)
    pred_curves = y_scaler.inverse_transform(pred_scaled)
    r2 = r2_score(Y_val_orig, pred_curves)
    print(f"  Temp {temp}: Validation R² = {r2:.3f}")
    results.append((temp,r2))
    
    # Save preprocessors
    joblib.dump(ohe, os.path.join(OUT_DIR,f"ohe_temp{temp}.joblib"))
    joblib.dump(pca, os.path.join(OUT_DIR,f"pca_temp{temp}.joblib"))
    joblib.dump(y_scaler, os.path.join(OUT_DIR,f"yscaler_temp{temp}.joblib"))

print("\nSummary R² per temperature:")
for t,r2 in results:
    print(f"  Temp {t}: {r2:.3f}")
