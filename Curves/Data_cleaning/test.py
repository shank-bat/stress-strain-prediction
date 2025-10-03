import pandas as pd

f = "SteelDataFinal/BCP325/BCP325_t22_tensile.csv"

# don’t force index_col
df = pd.read_csv(f, encoding="utf-8-sig")
print(df.head())
print("Columns:", df.columns.tolist())
