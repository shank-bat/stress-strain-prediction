import pandas as pd

# Load
df = pd.read_csv("/home/shu/projects/stress-strain-prediction/Keypoints/Databases/steel_strength.csv")

# Drop rows with missing targets
targets = ["yield strength", "tensile strength", "elongation"]
df_clean = df.dropna(subset=targets)

print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)

# Save cleaned dataset
df_clean.to_csv("/home/shu/projects/stress-strain-prediction/Keypoints/Databases/steel_strength_cleaned.csv", index=False)
print("Saved as steel_strength_cleaned.csv")
