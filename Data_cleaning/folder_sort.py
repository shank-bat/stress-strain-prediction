import pandas as pd
import shutil
from pathlib import Path

# Paths
base_dir = Path("Clean_Data")        # folder where you extracted the dataset
map_file = base_dir / "db_tag_clean_data_map.csv"
sorted_dir = Path("Sorted_Clean_Data")
sorted_dir.mkdir(exist_ok=True)

# Load the mapping
db_map = pd.read_csv(map_file, header=None)

# Define categories with keywords
categories = {
    "flange": ["flange"],
    "web": ["web"],
    "const": ["const"],
    "random": ["random"],
    "inc": ["inc"],
    "tensile": []  # default bucket for everything else
}

# Make category folders
for cat in categories:
    (sorted_dir / cat).mkdir(exist_ok=True)

# Classify function
def classify_file(path: str):
    for cat, keys in categories.items():
        for key in keys:
            if key.lower() in path.lower():
                return cat
    return "tensile"

# Sort and copy files
for _, row in db_map.iterrows():
    file_rel = row[1].lstrip("./")          # relative path in db_map
    file_abs = base_dir.parent / file_rel   # absolute path
    if file_abs.exists():
        cat = classify_file(file_rel)
        dest = sorted_dir / cat / Path(file_rel).name
        shutil.copy(file_abs, dest)

# Report counts
for cat in categories:
    count = len(list((sorted_dir / cat).glob("*.csv")))
    print(f"{cat}: {count} files")
