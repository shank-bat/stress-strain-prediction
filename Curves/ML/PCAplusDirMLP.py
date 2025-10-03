

#!/usr/bin/env python3
"""
Heavy PCA + MLP pipeline for stress-strain curves.

- Fixes OneHotEncoder compatibility issue (sparse vs sparse_output).
- Uses PCA to compress curves, MLP to predict PCA coefficients.
- Larger model, mixed precision if CUDA is available, scheduler, early stopping.
- Saves best model and sklearn objects (PCA, scalers, OHE).
- Plots multiple validation examples vs baselines.

Adjust hyperparams at the top.
"""

import os
import random
import math
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ---------- User params ----------
CSV_PATH = "/home/shu/projects/stress-strain-prediction/OutputData/aluminium_6061_curves.csv"
OUT_DIR = "/home/shu/projects/stress-strain-prediction/models_heavy"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_PCA_MAX = 30          # maximum PCA components to try (will be clipped to dataset size)
BATCH_SIZE = 32
EPOCHS = 600
LR = 2e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 40           # early stopping patience
GRAD_CLIP = 2.0
N_PLOT = 8              # number of val examples to plot
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 5
# ---------------------------------

# determinism
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ---------- Helpers ----------
def try_make_ohe(**kwargs):
    """Create OneHotEncoder robust to scikit-learn versions."""
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        # older versions used 'sparse'
        return OneHotEncoder(sparse=False, **kwargs)

def find_strain_columns(df):
    """Return sorted list of columns that are numeric strain grid (coerceable to float)."""
    cols = []
    for c in df.columns:
        try:
            _ = float(c)
            cols.append(c)
        except Exception:
            continue
    # sort by numeric value
    cols_sorted = sorted(cols, key=lambda x: float(x))
    return cols_sorted

# ---------- Load data ----------
print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("Rows in raw CSV:", len(df))

# Pivot so each specimen is one row
pivot = df.pivot_table(
    index=["specimen_type", "temperature", "lot", "specimen_id", "file_name"],
    columns="strain", values="stress"
).reset_index()

# identify strain columns robustly (column names might be floats or strings)
strain_cols = find_strain_columns(pivot)
if not strain_cols:
    # fallback: look for columns that look like numbers as strings
    strain_cols = [c for c in pivot.columns if re.match(r"^\d+(\.\d+)?$", str(c))]
    strain_cols = sorted(strain_cols, key=lambda x: float(x))

print(f"Found {len(strain_cols)} strain columns (grid points).")
strain_grid = np.array([float(c) for c in strain_cols])

# Build Y (N x M)
Y = pivot[strain_cols].values.astype(np.float32)   # may contain NaNs due to pivot

# Fill NaNs by interpolation along strain for each specimen
global_mean = np.nanmean(Y, axis=0)
for i in range(Y.shape[0]):
    row = Y[i]
    if np.any(np.isnan(row)):
        valid_mask = ~np.isnan(row)
        if valid_mask.sum() >= 2:
            xs = strain_grid[valid_mask]
            ys = row[valid_mask]
            Y[i] = np.interp(strain_grid, xs, ys)
        else:
            # fallback to global mean if we can't interpolate
            Y[i] = np.where(np.isnan(row), global_mean, row)

# ---------- Output scaling + PCA ----------
y_scaler = StandardScaler()
Y_scaled = y_scaler.fit_transform(Y)   # standardize each strain column

n_samples = Y.shape[0]
n_pca = min(N_PCA_MAX, n_samples - 1, Y.shape[1])
if n_pca < 2:
    n_pca = 2

pca = PCA(n_components=n_pca, svd_solver="full", random_state=SEED)
Y_pca = pca.fit_transform(Y_scaled)
print("PCA components:", n_pca)
print("Cumulative explained variance (first components):", np.cumsum(pca.explained_variance_ratio_)[:min(10, n_pca)])

# ---------- Metadata encoding ----------
X_meta = pivot[["specimen_type", "temperature", "lot"]].astype(str).reset_index(drop=True)
# One-hot encode specimen_type + lot robustly
ohe = try_make_ohe(handle_unknown="ignore")
cat = X_meta[["specimen_type", "lot"]]
cat_ohe = ohe.fit_transform(cat)   # guaranteed to be dense array by our try wrapper

# Temperature numeric scaling
temp = X_meta["temperature"].astype(float).values.reshape(-1, 1)
temp_scaler = StandardScaler()
temp_scaled = temp_scaler.fit_transform(temp)

X = np.hstack([cat_ohe.astype(np.float32), temp_scaled.astype(np.float32)])
print("Feature matrix shape:", X.shape)

# ---------- Train/val split (try stratify by temperature if possible) ----------
temps = X_meta["temperature"].astype(float).values
try:
    X_train, X_val, ytrain_pca, yval_pca, Y_train_orig, Y_val_orig = train_test_split(
        X, Y_pca, Y, test_size=0.2, random_state=SEED, stratify=temps
    )
except Exception:
    X_train, X_val, ytrain_pca, yval_pca, Y_train_orig, Y_val_orig = train_test_split(
        X, Y_pca, Y, test_size=0.2, random_state=SEED
    )

print("Train/val sizes:", len(X_train), len(X_val))

# ---------- PyTorch dataset ----------
class PCARegDataset(Dataset):
    def __init__(self, X, y_pca):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_pca, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = PCARegDataset(X_train, ytrain_pca)
val_ds = PCARegDataset(X_val, yval_pca)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Model ----------
class MLPBig(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.SiLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.15),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model = MLPBig(X.shape[1], n_pca).to(DEVICE)

# weight init
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
model.apply(init_weights)

criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)

use_amp = (DEVICE == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# ---------- Training loop ----------
best_val = float("inf")
pat = 0
best_path = os.path.join(OUT_DIR, "best_model.pt")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(xb)
            loss = criterion(pred, yb)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_ds)

    # validation
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb)
                vloss = criterion(pred, yb)
            running_val += vloss.item() * xb.size(0)
    val_loss = running_val / len(val_ds)
    scheduler.step(val_loss)

    if epoch % PRINT_EVERY == 0 or epoch == 1:
        print(f"Epoch {epoch:04d}  Train Loss: {train_loss:.6e}  Val Loss: {val_loss:.6e}  LR: {optimizer.param_groups[0]['lr']:.3e}")

    # early stopping + save
    if val_loss < best_val - 1e-12:
        best_val = val_loss
        pat = 0
        torch.save(model.state_dict(), best_path)
    else:
        pat += 1
        if pat >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best_val:.6e}")
            break

# load best
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
model.eval()

# ---------- Predict on val ----------
with torch.no_grad():
    Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    if use_amp:
        with torch.cuda.amp.autocast():
            pred_pca = model(Xv).cpu().numpy()
    else:
        pred_pca = model(Xv).cpu().numpy()

pred_Y_scaled = pca.inverse_transform(pred_pca)   # scaled Y
pred_Y = y_scaler.inverse_transform(pred_Y_scaled)

# baselines
global_mean_curve = np.mean(Y_train_orig, axis=0)
# per-temp mean from training set:
train_temps = (pivot.loc[train_ds.__len__()*0:]['temperature'].astype(float) 
               if False else None)  # avoid complex indexing here
# simpler per-temp mean: compute from pivot (but ensure only using training indices is better)
per_temp_mean = pivot.groupby("temperature")[strain_cols].mean()

# ---------- Numeric metrics ----------
mse_model = mean_squared_error(Y_val_orig, pred_Y)
mse_global = mean_squared_error(Y_val_orig, np.tile(global_mean_curve, (Y_val_orig.shape[0], 1)))
print(f"Val MSE - Model: {mse_model:.3f}   Global mean baseline: {mse_global:.3f}")

# save objects for later inference
joblib.dump(ohe, os.path.join(OUT_DIR, "ohe.joblib"))
joblib.dump(pca, os.path.join(OUT_DIR, "pca.joblib"))
joblib.dump(y_scaler, os.path.join(OUT_DIR, "y_scaler.joblib"))
joblib.dump(temp_scaler, os.path.join(OUT_DIR, "temp_scaler.joblib"))
torch.save(model.state_dict(), os.path.join(OUT_DIR, "final_model.pt"))
print("Saved model and preprocessing objects to", OUT_DIR)

# ---------- Plot several validation examples ----------
n_plot = min(N_PLOT, len(X_val))
idxs = np.random.choice(len(X_val), n_plot, replace=False)
fig, axes = plt.subplots(n_plot, 1, figsize=(7, 2.6 * n_plot), constrained_layout=True)
if n_plot == 1:
    axes = [axes]
for ax, idx in zip(axes, idxs):
    actual = Y_val_orig[idx]
    predicted = pred_Y[idx]
    # estimate temp to lookup per-temp mean (closest match)
    temp_norm = X_val[idx, -1]
    # invert temp scaler approximately:
    temp_orig = temp_scaler.inverse_transform(np.array([[temp_norm]]))[0,0]
    # pick per-temp mean or fallback
    try:
        temp_mean_curve = per_temp_mean.loc[float(temp_orig)].values
    except Exception:
        temp_mean_curve = global_mean_curve

    ax.plot(strain_grid, actual, label="Actual", lw=2)
    ax.plot(strain_grid, predicted, label="Predicted (MLP + PCA)", lw=1.8, linestyle="--")
    ax.plot(strain_grid, global_mean_curve, label="Global mean", lw=1, alpha=0.8)
    ax.plot(strain_grid, temp_mean_curve, label=f"Temp {temp_orig:.0f} mean", lw=1, alpha=0.8, linestyle=":")
    ax.set_ylabel("Stress (MPa)")
    ax.set_xlim(strain_grid.min(), strain_grid.max())
    ax.legend(fontsize=8, loc="upper right")
axes[-1].set_xlabel("Strain")
plt.suptitle("Validation: Actual vs Predicted vs Baselines", y=1.02)
plt.show()
# ---------- Verifications ---------
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Already have: pred_Y (val predictions), Y_val_orig (val truths)

# Mean squared error
mse_model = mean_squared_error(Y_val_orig, pred_Y)
mse_global = mean_squared_error(Y_val_orig, np.tile(global_mean_curve, (Y_val_orig.shape[0], 1)))

# R² score (accuracy-like metric)
r2 = r2_score(Y_val_orig, pred_Y)

# Normalized RMSE (percentage of stress range)
rmse = np.sqrt(mse_model)
stress_range = Y_val_orig.max() - Y_val_orig.min()
nrmse = rmse / stress_range * 100

print(f"Validation metrics:")
print(f"  MSE (Model): {mse_model:.3f}")
print(f"  MSE (Global mean): {mse_global:.3f}")
print(f"  R² Score: {r2:.4f} (1.0 = perfect)")
print(f"  Normalized RMSE: {nrmse:.2f}% of stress range")

# ---------- Quick tips ----------
print("\nNotes:")
print("- If model still underfits: increase n_pca or model capacity; if it overfits: reduce capacity, add dropout, or get more data.")
print("- If you can collect any per-specimen feature (composition, heat treatment flag, grain size proxy), add it to the metadata and re-run.")
print("- To run per-temperature models (often helpful), train separate models on each temp's subset (requires enough samples per temp).")
