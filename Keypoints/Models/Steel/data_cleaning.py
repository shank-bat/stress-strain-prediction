import pandas as pd
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]          # .../Keypoints/Models
DATA_PATH = BASE_DIR.parent / "Databases" / "steel_strength.csv"  # go up one (Keypoints/) then into Databases/
df = pd.read_csv(DATA_PATH)
# Drop rows with missing targets
targets = ["yield strength", "tensile strength", "elongation"]
df_clean = df.dropna(subset=targets)

print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)

# Save cleaned dataset
df_clean.to_csv("/home/shu/projects/stress-strain-prediction/Keypoints/Databases/steel_strength_cleaned.csv", index=False)
print("Saved as steel_strength_cleaned.csv")
