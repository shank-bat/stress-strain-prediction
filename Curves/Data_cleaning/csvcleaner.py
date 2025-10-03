import pandas as pd

# Load your data
df = pd.read_csv("/home/shu/projects/stress-strain-prediction/Curves/Databases/FinalEdited.csv")

# List of alloy/composition columns you expect
comp_cols = [
    "c","si","mn","p","s","cr","ni","mo","cu","v","nb","ti","al","b"
    # add others you have
]

# Ensure Fe is also in comp_cols for fill operations
all_cols = comp_cols + ["fe"]

# Fill NaNs in composition columns with 0
df[all_cols] = df[all_cols].fillna(0)

# Function to compute Fe as remainder (if Fe is zero)
def compute_fe(row):
    total_others = sum(row[elem] for elem in comp_cols)
    # If fe is zero or very small, we override
    if row["fe"] == 0 or abs(row["fe"] - (100 - total_others)) > 1e-6:
        return 100.0 - total_others
    else:
        return row["fe"]

df["fe"] = df.apply(compute_fe, axis=1)

# Define spec ranges you collected (for grades you have)
# Example structure: specs["S355"]["c"] = (0, 0.23) etc
specs = {
    "S355": {
        "c": (0.0, 0.23),
        "mn": (0.0, 1.60),
        "si": (0.0, 0.05),
        "p": (0.0, 0.05),
        "s": (0.0, 0.05),
    },
    "S235": {
        "c": (0.0, 0.22),
        "mn": (0.0, 1.60),
        "si": (0.0, 0.05),
        "p": (0.0, 0.05),
        "s": (0.0, 0.05),
    },
    "S275": {
        "c": (0.0, 0.25),
        "mn": (0.0, 1.60),
        "si": (0.0, 0.05),
        "p": (0.0, 0.04),
        "s": (0.0, 0.05),
    },
    "A500": {
        "c": (0.0, 0.26),
        "mn": (0.0, 1.35),
        "p": (0.0, 0.035),
        "s": (0.0, 0.035),
        "cu": (0.20, None),  # minimum copper
    },
    "S690": {
        "c": (0.10, 0.20),
        "mn": (0.0, 1.60),
        "si": (0.0, 0.50),
        "cr": (0.0, 0.30),
        "ni": (0.0, 0.30),
        "mo": (0.10, 0.15),
    }
    # You can add BCP325, HYP400 etc.
}

# Function to check violations on a row
def check_row_violation(row):
    grade = row["grade"]
    if grade not in specs:
        return []  # no spec to check
    vio = []
    for elem, (low, high) in specs[grade].items():
        val = row.get(elem, 0.0)
        # If only low or only high defined, handle None
        if low is not None and val < low - 1e-6:
            vio.append((elem, val, "< lower bound", low))
        if high is not None and val > high + 1e-6:
            vio.append((elem, val, "> upper bound", high))
    return vio

# Collect all violations
violations = []
for idx, row in df.iterrows():
    vio = check_row_violation(row)
    if vio:
        for (elem, val, relation, bound) in vio:
            violations.append({
                "row": idx,
                "grade": row["grade"],
                "element": elem,
                "value": val,
                "relation": relation,
                "bound": bound
            })

violations_df = pd.DataFrame(violations)

# Output or inspect
print("Total violations:", len(violations))
print(violations_df.head(20))

# Optionally, save to CSV
violations_df.to_csv("composition_violations.csv", index=False)
