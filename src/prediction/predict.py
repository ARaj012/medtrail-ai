"""
One prediction interface for all four MedTrail disease models.

Input:  a patient profile and a list of test results in the team's data-contract format.
Output: for each disease, either a risk estimate or a message saying which values are needed.
Every value used keeps the report_id it came from, so each result can be traced to its source.

Run a demo from the project folder:
    python -m src.prediction.predict
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

DISCLAIMER = "Screening estimate from a small public dataset. Not a diagnosis."

# A few common alternative names -> MedTrail standard test name.
# (Member 4's OCR service does the full normalisation; this is a safety net.)
ALIASES = {
    "hb": "Hemoglobin", "hgb": "Hemoglobin", "haemoglobin": "Hemoglobin",
    "creatinine": "Serum Creatinine", "s. creatinine": "Serum Creatinine",
    "urea": "Blood Urea", "alt": "SGPT", "ast": "SGOT", "alp": "Alkaline Phosphatase",
    "fbs": "Fasting Blood Sugar", "cholesterol": "Total Cholesterol",
    "bilirubin total": "Total Bilirubin", "bilirubin direct": "Direct Bilirubin",
}

# For each model: lab inputs (model column -> (standard test name, expected unit)),
# profile inputs (model column -> profile field), and a short reliability note.
MODEL_SPECS = {
    "diabetes": {
        "label": "Type 2 diabetes",
        "tests": {"Glucose": ("Glucose", "mg/dL")},
        "profile": {"BloodPressure": "diastolic_bp", "BMI": "bmi", "Age": "age"},
        "note": "Trained on 768 women of Pima heritage aged 21+; less reliable for other groups.",
    },
    "kidney": {
        "label": "Chronic kidney disease",
        "tests": {
            "sc": ("Serum Creatinine", "mg/dL"), "bu": ("Blood Urea", "mg/dL"),
            "hemo": ("Hemoglobin", "g/dL"), "al": ("Urine Albumin", None),
            "su": ("Urine Sugar", None), "bgr": ("Glucose", "mg/dL"),
        },
        "profile": {"bp": "diastolic_bp", "age": "age"},
        "note": "Trained on 400 patients from one hospital in Tamil Nadu; the dataset is easy, "
                "so its very high scores overstate real-world accuracy.",
    },
    "liver": {
        "label": "Liver disease",
        "tests": {
            "TB": ("Total Bilirubin", "mg/dL"), "DB": ("Direct Bilirubin", "mg/dL"),
            "Sgpt": ("SGPT", "U/L"), "Sgot": ("SGOT", "U/L"),
            "Alkphos": ("Alkaline Phosphatase", "U/L"), "TP": ("Total Protein", "g/dL"),
            "ALB": ("Albumin", "g/dL"), "A/G Ratio": ("A/G Ratio", None),
        },
        "profile": {"Age": "age", "Sex": "sex"},
        "note": "Indicative only: moderate accuracy (ROC-AUC about 0.76) on 570 patients "
                "from Andhra Pradesh.",
    },
    "heart": {
        "label": "Heart disease",
        "tests": {"chol": ("Total Cholesterol", "mg/dL"), "fbs": ("Fasting Blood Sugar", "mg/dL")},
        "profile": {"age": "age", "sex": "sex", "trestbps": "systolic_bp"},
        "note": "Indicative only: uses 5 of 13 inputs (no ECG or exercise test), ROC-AUC about 0.70, "
                "trained on 303 US patients.",
    },
}

# Inputs that need converting before the model can use them
CONVERSIONS = {
    ("heart", "fbs"): lambda value: int(value > 120),   # fasting sugar above 120 mg/dL -> 1
}


def risk_level(probability):
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Moderate"
    return "High"


@lru_cache(maxsize=None)
def load_model(disease):
    """Load a saved pipeline once and keep it in memory."""
    return joblib.load(MODELS_DIR / f"{disease}_pipeline.joblib")


@lru_cache(maxsize=None)
def load_metrics(disease):
    return json.loads((MODELS_DIR / f"{disease}_metrics.json").read_text())


def standard_name(test_name):
    name = str(test_name).strip()
    return ALIASES.get(name.lower(), name)


def same_unit(a, b):
    return str(a).replace(" ", "").lower() == str(b).replace(" ", "").lower()


def latest_results(test_results):
    """Keep only the most recent result of each test (dates are YYYY-MM-DD, so text order works)."""
    latest = {}
    for result in sorted(test_results, key=lambda r: str(r.get("test_date", ""))):
        if result.get("value") is None:
            continue
        latest[standard_name(result["test_name"])] = result
    return latest


def profile_values(profile):
    """Profile fields the models use, including BMI worked out from height and weight."""
    values = {
        "age": profile.get("age"),
        "systolic_bp": profile.get("systolic_bp"),
        "diastolic_bp": profile.get("diastolic_bp"),
    }
    sex = str(profile.get("sex", "")).strip().lower()
    values["sex"] = {"male": 1, "m": 1, "female": 0, "f": 0}.get(sex)
    height_cm, weight_kg = profile.get("height_cm"), profile.get("weight_kg")
    if height_cm and weight_kg:
        values["bmi"] = round(weight_kg / (height_cm / 100) ** 2, 1)
    return values


def build_inputs(disease, profile, latest):
    """Collect one model's inputs. Returns (row, inputs_used, missing)."""
    spec = MODEL_SPECS[disease]
    row, used, missing = {}, {}, []

    for column, (test_name, unit) in spec["tests"].items():
        result = latest.get(test_name)
        if result is None:
            missing.append(test_name)
            continue
        if unit and result.get("unit") and not same_unit(result["unit"], unit):
            missing.append(f"{test_name} in {unit} (report gives {result['unit']})")
            continue
        value = float(result["value"])
        convert = CONVERSIONS.get((disease, column))
        row[column] = convert(value) if convert else value
        used[column] = {"test_name": test_name, "value": value, "unit": result.get("unit"),
                        "test_date": result.get("test_date"), "report_id": result.get("report_id")}
        if convert:
            used[column]["model_value"] = row[column]     # e.g. fasting sugar 132 -> fbs = 1

    values = profile_values(profile)
    for column, field in spec["profile"].items():
        if values.get(field) is None:
            missing.append({"bmi": "height and weight"}.get(field, field.replace("_", " ")))
            continue
        row[column] = values[field]
        used[column] = {"profile_field": field, "value": values[field]}

    return row, used, missing


def predict_disease(disease, profile, test_results):
    """Risk estimate for one disease, or a clear list of what is needed."""
    spec = MODEL_SPECS[disease]
    latest = latest_results(test_results)
    row, used, missing = build_inputs(disease, profile, latest)
    base = {"disease": disease, "label": spec["label"]}

    if missing:
        return {**base, "status": "needs_more_data", "missing": missing,
                "message": "Needs " + ", ".join(missing) + "."}

    try:
        model = load_model(disease)
    except FileNotFoundError:
        return {**base, "status": "model_unavailable",
                "message": f"The {disease} model file is missing from the models folder."}

    features = list(model.feature_names_in_)            # same column order as in training
    X = pd.DataFrame([row])[features]
    probability = float(model.predict_proba(X)[0, 1])
    return {
        **base,
        "status": "ok",
        "probability": round(probability, 3),
        "risk_level": risk_level(probability),
        "model": load_metrics(disease)["model"],
        "inputs_used": used,
        "source_report_ids": sorted({v["report_id"] for v in used.values() if v.get("report_id")}),
        "note": spec["note"],
        "disclaimer": DISCLAIMER,
    }


def predict_all(profile, test_results):
    """Run every disease model on one patient."""
    return {disease: predict_disease(disease, profile, test_results) for disease in MODEL_SPECS}


if __name__ == "__main__":
    demo_profile = {"age": 52, "sex": "female", "height_cm": 158, "weight_kg": 78,
                    "systolic_bp": 148, "diastolic_bp": 92}

    def result(rid, name, value, unit, date):
        return {"report_id": rid, "patient_id": "P001", "test_name": name, "value": value,
                "unit": unit, "ref_low": None, "ref_high": None, "test_date": date,
                "lab_name": "Demo Lab"}

    demo_results = [
        result("R100", "Glucose", 150, "mg/dL", "2026-06-10"),
        result("R101", "Glucose", 168, "mg/dL", "2026-10-20"),          # latest glucose is used
        result("R101", "Serum Creatinine", 1.1, "mg/dL", "2026-10-20"),
        result("R101", "Blood Urea", 32, "mg/dL", "2026-10-20"),
        result("R101", "Hb", 11.8, "g/dL", "2026-10-20"),                # alias -> Hemoglobin
        result("R101", "Urine Albumin", 1, "", "2026-10-20"),
        result("R101", "Urine Sugar", 1, "", "2026-10-20"),
        result("R102", "Total Cholesterol", 236, "mg/dL", "2026-10-22"),
        result("R102", "Fasting Blood Sugar", 132, "mg/dL", "2026-10-22"),
    ]
    print(json.dumps(predict_all(demo_profile, demo_results), indent=2))