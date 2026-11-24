"""
Explore the Synthea sample: which lab tests and diagnoses it contains, and how many
patients have repeated results (trend analysis needs at least 3 results over time).

Run from the project folder:
    python src/history_analysis/explore_synthea.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "datasets" / "synthetic_patients" / "raw" / "csv"

# Words to look for in lab / vital-sign names (observations.csv), lower case
LAB_KEYWORDS = [
    "a1c", "glucose", "creatinine", "urea nitrogen", "hemoglobin", "glomerular",
    "albumin", "aminotransferase", "bilirubin", "blood pressure", "cholesterol",
]

# Words to look for in diagnosis names (conditions.csv), lower case
CONDITION_KEYWORDS = ["diabetes", "kidney", "liver", "hepat", "anemia", "hypertension"]


def explore_labs():
    """For each keyword: the matching test names, rows, patients, and patients with 3+ results."""
    obs = pd.read_csv(CSV / "observations.csv", usecols=["DATE", "PATIENT", "DESCRIPTION", "UNITS"])
    print(f"observations.csv: {len(obs)} rows, {obs['PATIENT'].nunique()} patients")
    names = obs["DESCRIPTION"].fillna("").str.lower()
    for word in LAB_KEYWORDS:
        match = obs[names.str.contains(word, regex=False)]
        if match.empty:
            print(f"\n[{word}] NOT FOUND")
            continue
        summary = match.groupby("DESCRIPTION").agg(
            rows=("PATIENT", "size"),
            patients=("PATIENT", "nunique"),
            unit=("UNITS", "first"),
        )
        results_per_patient = match.groupby(["DESCRIPTION", "PATIENT"]).size()
        summary["patients_3plus"] = (results_per_patient >= 3).groupby(level="DESCRIPTION").sum()
        summary.index = summary.index.str.slice(0, 55)  # shorten very long test names
        print(f"\n[{word}]")
        print(summary.sort_values("rows", ascending=False).head(3).to_string())


def explore_conditions():
    """For each keyword: the matching diagnosis names and how many patients have them."""
    cond = pd.read_csv(CSV / "conditions.csv", usecols=["START", "PATIENT", "DESCRIPTION"])
    names = cond["DESCRIPTION"].fillna("").str.lower()
    for word in CONDITION_KEYWORDS:
        match = cond[names.str.contains(word, regex=False)]
        if match.empty:
            print(f"\n[{word}] NOT FOUND")
            continue
        counts = match.groupby("DESCRIPTION")["PATIENT"].nunique().sort_values(ascending=False)
        print(f"\n[{word}] patients per diagnosis")
        print(counts.head(5).to_string())


def explore_meds_and_allergies():
    meds = pd.read_csv(CSV / "medications.csv", usecols=["PATIENT"])
    allergies = pd.read_csv(CSV / "allergies.csv", usecols=["PATIENT"])
    print(f"\nPatients with medication records: {meds['PATIENT'].nunique()}")
    print(f"Patients with allergy records:    {allergies['PATIENT'].nunique()}")


if __name__ == "__main__":
    print("===== LAB TESTS AND VITAL SIGNS =====")
    explore_labs()
    print("\n===== DIAGNOSES =====")
    explore_conditions()
    print("\n===== MEDICATIONS AND ALLERGIES =====")
    explore_meds_and_allergies()