"""Data preparation: clean the dataset, split into train/test and save locally."""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

df = pd.read_csv(DATA_PATH)

# Drop the unnamed index column and the identifier - they carry no predictive signal
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == ""] + ["CustomerID"])

# Standardise inconsistent category labels
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Remove duplicates and impute any missing values (median for numeric, mode for categorical)
df = df.drop_duplicates()
for col in df.columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median() if df[col].dtype != "object" else df[col].mode()[0])

X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print(f"Cleaned data: {df.shape}. Train: {Xtrain.shape}, Test: {Xtest.shape}")
print(f"Train positive rate: {ytrain.mean():.3f} | Test positive rate: {ytest.mean():.3f}")
