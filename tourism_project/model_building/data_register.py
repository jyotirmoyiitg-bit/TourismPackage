"""Data registration: validate the dataset stored in the repo and print a summary."""
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

df = pd.read_csv(DATA_PATH)
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    print(f"VALIDATION FAILED - missing columns: {missing}")
    sys.exit(1)

print(f"Dataset registered from {DATA_PATH}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Extra columns (will be dropped in prep): {[c for c in df.columns if c not in EXPECTED_COLUMNS]}")
print("\nData types:\n", df.dtypes.to_string())
print("\nMissing values per column:\n", df.isna().sum().to_string())
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print("\nTarget distribution (ProdTaken):\n", df["ProdTaken"].value_counts(normalize=True).round(3).to_string())
print("\nValidation passed.")
