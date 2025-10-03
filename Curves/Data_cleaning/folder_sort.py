import shutil
from pathlib import Path

# Point to your Clean_Data folder
base_dir = Path("Clean_Data")   # adjust if needed

# Where to save
out_root = Path("Selected_Tensile")
if out_root.exists():
    shutil.rmtree(out_root)
out_root.mkdir(parents=True, exist_ok=True)

# Keywords to exclude (non-tensile)
exclude_keywords = ["flange", "web", "const", "random", "inc"]

def is_lp1(path_str: str):
    return "lp1" in path_str

# Collect all csv/pdf files
all_files = list(base_dir.rglob("*"))
selected = []

for p in all_files:
    if p.suffix.lower() not in [".csv", ".pdf"]:
        continue
    pstr = str(p).lower()
    if any(ex in pstr for ex in exclude_keywords):
        continue
    # Accept if file is LP1 OR contains tensile OR plates OR base metal
    if is_lp1(pstr) or "tensile" in pstr or "plate" in pstr or "base metal" in pstr:
        selected.append(p)

# Copy to output
for p in selected:
    dest = out_root / p.name
    shutil.copy(p, dest)

print(f"Copied {len(selected)} files into {out_root}")
