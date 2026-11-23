"""
Download the four disease datasets into datasets/raw/.

- Kidney (CKD), liver and heart come from the UCI Machine Learning Repository.
- Diabetes (Pima) comes from OpenML dataset 37, a copy of the original UCI file,
  because the Kaggle copy is no longer available. If a teammate gives you their
  diabetes.csv, put it in datasets/raw/diabetes/ first and it will be used instead.

Run from the project folder:
    python src/preprocessing/download_datasets.py
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo

# Project folder = two levels above this file (src/preprocessing/ -> medtrail-ai/)
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets" / "raw"

# folder name -> (UCI dataset id, file name to save, target column)
UCI_DATASETS = {
    "kidney": (336, "ckd.csv", "class"),
    "liver": (225, "ilpd.csv", "Selector"),
    "heart": (45, "heart_cleveland.csv", "num"),
}

# OpenML short names -> the standard column names used by the team's diabetes model
PIMA_COLUMNS = {
    "preg": "Pregnancies",
    "plas": "Glucose",
    "pres": "BloodPressure",
    "skin": "SkinThickness",
    "insu": "Insulin",
    "mass": "BMI",
    "pedi": "DiabetesPedigreeFunction",
    "age": "Age",
}


def summarise(name, df, target):
    """Print size, missing values and target counts for one dataset."""
    print(f"\n=== {name.upper()} ===")
    print(f"Rows: {len(df)}   Columns: {df.shape[1]}")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("Missing values:", "none" if missing.empty else missing.to_dict())
    if target in df.columns:
        print(f"Target '{target}':", df[target].value_counts(dropna=False).to_dict())
    else:
        print(f"WARNING: target '{target}' not found. Columns: {list(df.columns)}")


def download_uci():
    """Download each UCI dataset and save the data plus its column dictionary."""
    for name, (uci_id, filename, target) in UCI_DATASETS.items():
        folder = RAW / name
        folder.mkdir(parents=True, exist_ok=True)
        try:
            repo = fetch_ucirepo(id=uci_id)
        except Exception as error:
            print(f"\n=== {name.upper()} === DOWNLOAD FAILED (UCI id {uci_id}): {error}")
            continue
        df = repo.data.original
        df.to_csv(folder / filename, index=False)
        # Column dictionary: name, role, type, description, units, missing values
        repo.variables.to_csv(folder / f"{Path(filename).stem}_columns.csv", index=False)
        summarise(name, df, target)


def download_pima():
    """Get the Pima diabetes data from OpenML (id 37) unless a copy already exists."""
    folder = RAW / "diabetes"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "diabetes.csv"
    if path.exists():
        print(f"\n=== DIABETES === using the existing file: {path}")
    else:
        try:
            frame = fetch_openml(data_id=37, as_frame=True).frame
        except Exception as error:
            print(f"\n=== DIABETES === DOWNLOAD FAILED (OpenML id 37): {error}")
            return
        df = frame.rename(columns=PIMA_COLUMNS)
        # Target: tested_positive -> 1, tested_negative -> 0
        df["Outcome"] = (df.pop("class").astype(str) == "tested_positive").astype(int)
        whole_numbers = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "Age"]
        df[whole_numbers] = df[whole_numbers].astype(int)
        df.to_csv(path, index=False)
        print("\n(Diabetes data downloaded from OpenML dataset 37, originally from UCI.)")

    df = pd.read_csv(path)
    summarise("diabetes", df, "Outcome")
    # In this dataset a value of 0 for these measurements is impossible: it means "not recorded"
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    zeros = {col: int((df[col] == 0).sum()) for col in zero_cols if col in df.columns}
    print("Zeros that really mean 'missing':", zeros)


if __name__ == "__main__":
    download_uci()
    download_pima()
    print(f"\nFiles are in: {RAW}")