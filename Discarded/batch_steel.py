import os
import pandas as pd

base_path = "/home/shu/projects/stress-strain-prediction/Clean_Data/"

all_tensile = []

for root, dirs, files in os.walk(base_path):
    for f in files:
        if f.endswith(".csv") and "tensile" in f.lower():
            file_path = os.path.join(root, f)
            try:
                df = pd.read_csv(file_path)
                
                # only keep Sigma_true and e_true if they exist
                cols_to_keep = [c for c in ["Sigma_true", "e_true"] if c in df.columns]
                if not cols_to_keep:
                    print(f"Skipped {file_path}: no Sigma_true/e_true found")
                    continue
                
                df = df[cols_to_keep].copy()
                
                # add metadata
                df["file_name"] = f
                df["steel_grade"] = os.path.basename(os.path.dirname(file_path))
                df["full_path"] = file_path
                
                all_tensile.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# merge all into one dataframe
if all_tensile:
    df_tensile = pd.concat(all_tensile, ignore_index=True)
    out_path = os.path.join(base_path, "tensile_curves_sigma_e.csv")
    df_tensile.to_csv(out_path, index=False)
    print(f"Saved clean tensile dataset to {out_path}")
    print(df_tensile.head())
else:
    print("No tensile data found.")
