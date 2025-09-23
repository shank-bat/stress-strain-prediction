import os
import pandas as pd

root_dir = "/home/shu/Projects/ss_curve/Clean_Data/"

all_data = []

def normalize_columns(df):
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    # stress
    stress_candidates = [c for c in cols if "stress" in c or "sigma" in c]
    # strain
    strain_candidates = [c for c in cols if "strain" in c or "e_" in c]

    if not stress_candidates or not strain_candidates:
        raise ValueError(f"No stress/strain columns found in {df.columns}")

    stress_col = stress_candidates[0]
    strain_col = strain_candidates[0]

    out = pd.DataFrame({
        "stress": df[stress_col],
        "strain": df[strain_col]
    })

    return out

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if not file.endswith(".csv") or file == "db_tag_clean_data_map.csv":
            continue

        file_path = os.path.join(subdir, file)
        try:
            df = pd.read_csv(file_path)
        except Exception:
            try:
                df = pd.read_csv(file_path, delimiter=";")
            except Exception:
                try:
                    df = pd.read_csv(file_path, delimiter="\t")
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
                    continue

        try:
            df_norm = normalize_columns(df)
            grade = os.path.basename(subdir)
            df_norm["steel_grade"] = grade
            df_norm["file_name"] = file
            all_data.append(df_norm)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

if all_data:
    master_df = pd.concat(all_data, ignore_index=True)
    out_path = os.path.join(root_dir, "all_steel_curves.csv")
    master_df.to_csv(out_path, index=False)
    print(f"Saved {len(master_df)} rows to {out_path}")
else:
    print("No usable CSVs found.")
