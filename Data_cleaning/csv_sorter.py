import pandas as pd
from pathlib import Path

# === CONFIG ===
data_root = Path("SteelDataFinal")         # folder with your curated tensile subfolders
summary_csv = data_root / "Overall_summary.csv"
out_csv = "Merged_Tensile_Composition.csv"

# === LOAD SUMMARY ===
summary = pd.read_csv(summary_csv)

# Drop flanges/webs
mask = ~summary["source"].str.contains("flange|web", case=False, na=False)
summary = summary[mask]

# Keep only useful columns
cols_keep = [
    "grade", "spec", "source",
    "gage_length__mm_", "avg_reduced_dia__mm_",
    "yield_stress__mpa_", "elastic_modulus__mpa_", "fu_n__mpa_", "fracture_strain",
    # composition features
    "c","si","mn","p","s","n","cu","mo","ni","cr","v","nb","ti","al","b","zr","sn","ca","h","fe",
    "file"
]
summary = summary[cols_keep]

# Normalize the 'file' column to just the filename
summary["file"] = summary["file"].apply(lambda x: Path(x).name if isinstance(x,str) else x)

# === LOAD ALL CURATED STRESS–STRAIN FILES ===
all_data = []

for f in data_root.rglob("*.csv"):
    fname = f.name.lower()
    if fname.startswith("overall") or "yield_stress" in fname or "db_tag" in fname:
        continue  # skip summary/extra files

    try:
        df = pd.read_csv(f, encoding="utf-8-sig")
    except:
        df = pd.read_csv(f)

    # Drop dummy index columns if present
    if df.columns[0].lower().startswith("unnamed") or df.columns[0].strip() == "":
        df = df.drop(df.columns[0], axis=1)

    # Clean column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Normalize variants
    if "Sigma_true" not in df.columns:
        if "sigma_true" in df.columns:
            df.rename(columns={"sigma_true": "Sigma_true"}, inplace=True)
        elif "Eng_Stress" in df.columns:
            df.rename(columns={"Eng_Stress": "Sigma_true"}, inplace=True)
        elif "eng_stress" in df.columns:
            df.rename(columns={"eng_stress": "Sigma_true"}, inplace=True)

    if "e_true" not in df.columns:
        if "E_true" in df.columns:
            df.rename(columns={"E_true": "e_true"}, inplace=True)
        elif "Eng_Strain" in df.columns:
            df.rename(columns={"Eng_Strain": "e_true"}, inplace=True)
        elif "eng_strain" in df.columns:
            df.rename(columns={"eng_strain": "e_true"}, inplace=True)
        elif "strain" in df.columns:
            df.rename(columns={"strain": "e_true"}, inplace=True)

    # Debug if still missing
    if "Sigma_true" not in df.columns or "e_true" not in df.columns:
        print(f"⚠️  {f.name} missing stress/strain after normalization → Columns: {list(df.columns)}")

    df["file"] = f.name
    all_data.append(df)

# === MERGE ALL ===
stress_strain = pd.concat(all_data, ignore_index=True)
merged = stress_strain.merge(summary, on="file", how="left")

# === SAVE ===
merged.to_csv(out_csv, index=False)
print(f"✅ Saved merged dataset with {len(merged)} rows to {out_csv}")
