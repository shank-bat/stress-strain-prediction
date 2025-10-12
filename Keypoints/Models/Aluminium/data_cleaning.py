import pandas as pd
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]          # .../Keypoints/Models
DATA_PATH = BASE_DIR.parent / "Databases" / "al_data.csv"  # go up one (Keypoints/) then into Databases/
df = pd.read_csv(DATA_PATH)
# Drop useless index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Define targets
targets = ["Elongation (%)", "Tensile Strength (MPa)", "Yield Strength (MPa)"]

# Drop rows with missing values in targets
df_clean = df.dropna(subset=targets)

print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)

# Save cleaned dataset if needed
df_clean.to_csv("/home/shu/projects/stress-strain-prediction/Keypoints/Databases/al_data_cleaned.csv", index=False)
