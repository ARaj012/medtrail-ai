"""
Global SHAP importance for every saved disease model.

For each model: average size of each input's SHAP contribution over all patients in its
dataset. Saves models/<disease>_global_importance.json (for the dashboard) and
docs/figures/shap_global_<disease>.png (for the report).

Run from the project folder:
    python -m src.explainability.global_importance
"""
import json

import matplotlib
matplotlib.use("Agg")                      # save charts to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.explainability.explain import global_importance
from src.prediction.predict import MODEL_SPECS, ROOT, input_labels, load_model

RAW = ROOT / "datasets" / "raw"
FIGURES = ROOT / "docs" / "figures"


def load_dataset(disease):
    """Load each dataset with the same cleaning used in its training notebook."""
    if disease == "diabetes":
        df = pd.read_csv(RAW / "diabetes" / "diabetes.csv")
        columns = ["Glucose", "BloodPressure", "BMI"]
        df[columns] = df[columns].replace(0, np.nan)
    elif disease == "kidney":
        df = pd.read_csv(RAW / "kidney" / "ckd.csv")
        for column in ["sc", "bu", "hemo", "al", "su", "bgr", "bp", "age"]:
            df[column] = pd.to_numeric(df[column].astype(str).str.strip(), errors="coerce")
    elif disease == "liver":
        df = pd.read_csv(RAW / "liver" / "ilpd.csv").drop_duplicates()
        df["Sex"] = df["Gender"].astype(str).str.strip().str.lower().map({"male": 1, "female": 0})
    else:
        df = pd.read_csv(RAW / "heart" / "heart_cleveland.csv")
    return df


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for disease, spec in MODEL_SPECS.items():
        model = load_model(disease)
        X = load_dataset(disease)[list(model.feature_names_in_)]
        importance, method, unit = global_importance(model, X)
        labels = input_labels(disease)

        record = {
            "disease": disease, "method": method, "unit": unit, "patients": len(X),
            "importance": [{"input": col, "name": labels.get(col, col), "mean_abs_shap": round(float(v), 4)}
                           for col, v in importance.items()],
        }
        (ROOT / "models" / f"{disease}_global_importance.json").write_text(json.dumps(record, indent=2))

        names = [labels.get(col, col) for col in importance.index][::-1]   # biggest at the top
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.2))
        ax.barh(names, importance.values[::-1], color="#2a78d6", height=0.6)
        ax.set_xlabel(f"Average |SHAP contribution| ({unit})")
        ax.set_title(f"What drives the {spec['label'].lower()} model ({len(X)} patients)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES / f"shap_global_{disease}.png", dpi=120)
        plt.close(fig)

        top = ", ".join(f"{labels.get(c, c)} {v:.3f}" for c, v in importance.head(3).items())
        print(f"{disease:<9} {method} [{unit}] top inputs: {top}")
    print("Saved JSON files in models/ and charts in docs/figures/")


if __name__ == "__main__":
    main()