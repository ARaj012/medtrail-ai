"""
Build MedTrail medical-history files for the 3 synthetic demonstration patients.

Reads the Synthea CSV sample and writes one JSON file per patient to
datasets/synthetic_patients/. Each file follows the proposed history format
(docs/history_format.md): profile, reports, test_results (team data contract),
conditions, medications, allergies and blood-pressure readings. Every item
keeps the report_id it came from.

Run from the project folder:
    python -m src.history_analysis.build_history
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "datasets" / "synthetic_patients" / "raw" / "csv"
OUT = ROOT / "datasets" / "synthetic_patients"

# label -> (Synthea patient id prefix, short description)
DEMO_PATIENTS = {
    "A": ("56cfe6a5", "diabetes"),
    "B": ("cb328021", "kidney"),
    "C": ("58eb83f7", "liver"),
}
YEARS_OF_HISTORY = 5      # keep the last 5 years before the latest lab report
REPORTS_PER_TEST = 3      # for every test, keep the latest 3 reports that contain it

# Synthea test name -> (MedTrail standard name, unit, typical adult reference low, high).
# Synthea gives no reference ranges, so these typical adult ranges are added for the demo.
TEST_MAP = {
    "Hemoglobin A1c/Hemoglobin.total in Blood": ("HbA1c", "%", 4.0, 5.6),
    "Glucose": ("Glucose", "mg/dL", 70, 99),
    "Glucose [Mass/volume] in Serum or Plasma": ("Glucose", "mg/dL", 70, 99),
    "Creatinine": ("Serum Creatinine", "mg/dL", 0.6, 1.2),
    "Creatinine [Mass/volume] in Serum or Plasma": ("Serum Creatinine", "mg/dL", 0.6, 1.2),
    "Urea Nitrogen": ("Blood Urea Nitrogen", "mg/dL", 7, 20),
    "Urea nitrogen [Mass/volume] in Serum or Plasma": ("Blood Urea Nitrogen", "mg/dL", 7, 20),
    "Glomerular filtration rate/1.73 sq M.predicted": ("eGFR", "mL/min/1.73m2", 60, None),
    "Estimated Glomerular Filtration Rate": ("eGFR", "mL/min/1.73m2", 60, None),
    "Microalbumin Creatinine Ratio": ("Urine Albumin/Creatinine Ratio", "mg/g", None, 30),
    "Hemoglobin [Mass/volume] in Blood": ("Hemoglobin", "g/dL", None, None),   # range set by sex below
    "Sodium": ("Sodium", "mmol/L", 135, 145),
    "Potassium": ("Potassium", "mmol/L", 3.5, 5.1),
    "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma": ("SGPT", "U/L", 7, 40),
    "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma": ("SGOT", "U/L", 8, 40),
    "Bilirubin.total [Mass/volume] in Serum or Plasma": ("Total Bilirubin", "mg/dL", 0.2, 1.2),
    "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma": ("Alkaline Phosphatase", "U/L", 44, 147),
    "Protein [Mass/volume] in Serum or Plasma": ("Total Protein", "g/dL", 6.0, 8.3),
    "Albumin [Mass/volume] in Serum or Plasma": ("Albumin", "g/dL", 3.5, 5.0),
    "Total Cholesterol": ("Total Cholesterol", "mg/dL", None, 200),
    "Low Density Lipoprotein Cholesterol": ("LDL Cholesterol", "mg/dL", None, 100),
    "High Density Lipoprotein Cholesterol": ("HDL Cholesterol", "mg/dL", 40, None),
    "Triglycerides": ("Triglycerides", "mg/dL", None, 150),
}
HEMOGLOBIN_RANGE = {"male": (13.5, 17.5), "female": (12.0, 15.5)}
NOT_CONDITIONS = r"\((?:finding|situation|person)"   # Synthea social/administrative entries


def load_csv(name, columns=None):
    return pd.read_csv(CSV / f"{name}.csv", usecols=columns, low_memory=False)


def day(timestamp):
    return str(timestamp)[:10] if pd.notna(timestamp) else None


def build_patient(label, prefix, focus, data):
    patients, encounters, observations, conditions, medications, allergies, organisations = data
    person = patients[patients["Id"].str.startswith(prefix)].iloc[0]
    pid = person["Id"]
    sex = {"M": "male", "F": "female"}[person["GENDER"]]

    obs = observations[observations["PATIENT"] == pid].copy()
    obs["day"] = obs["DATE"].str[:10]
    labs = obs[obs["DESCRIPTION"].isin(TEST_MAP)].copy()
    labs["value"] = pd.to_numeric(labs["VALUE"], errors="coerce")
    labs = labs.dropna(subset=["value"])

    last_day = pd.Timestamp(labs["day"].max())
    first_day = (last_day - pd.DateOffset(years=YEARS_OF_HISTORY)).strftime("%Y-%m-%d")
    labs = labs[labs["day"] >= first_day]
    labs["test"] = labs["DESCRIPTION"].map(lambda d: TEST_MAP[d][0])

    # Synthea stores most total-protein values in g/L under a g/dL label: convert those to g/dL
    cleaning_notes = []
    wrong_unit = (labs["test"] == "Total Protein") & (labs["value"] > 20)
    if wrong_unit.any():
        labs.loc[wrong_unit, "value"] = labs.loc[wrong_unit, "value"] / 10
        cleaning_notes.append(f"{int(wrong_unit.sum())} Total Protein value(s) were in g/L although labelled g/dL "
                              "in Synthea; they were divided by 10.")

    # For every test keep the latest REPORTS_PER_TEST reports containing it (so every test has a trend)
    keep = set()
    for _, group in labs.groupby("test"):
        keep |= set(group.groupby("ENCOUNTER")["day"].min().sort_values().tail(REPORTS_PER_TEST).index)
    lab_encounters = labs[labs["ENCOUNTER"].isin(keep)].groupby("ENCOUNTER")["day"].min().sort_values()
    labs = labs[labs["ENCOUNTER"].isin(lab_encounters.index)]

    # Conditions, medications and allergies recorded for this patient
    cond = conditions[(conditions["PATIENT"] == pid)
                      & ~conditions["DESCRIPTION"].str.contains(NOT_CONDITIONS, regex=True)]
    cond = cond[cond["STOP"].isna() | (cond["STOP"].astype(str).str[:10] >= first_day)]
    cond = cond.sort_values("START").drop_duplicates("DESCRIPTION")

    # Synthea repeats a medicine at every refill: keep one row per medicine that was taken in the
    # window, with its true first start date (and the visit where it was first prescribed)
    all_meds = medications[medications["PATIENT"] == pid].sort_values("START")
    meds = all_meds[all_meds["STOP"].isna() | (all_meds["STOP"].astype(str).str[:10] >= first_day)]
    med_rows = []
    for name, group in meds.groupby("DESCRIPTION", sort=False):
        first = all_meds[all_meds["DESCRIPTION"] == name].iloc[0]
        ongoing = group["STOP"].isna().any()
        med_rows.append({"name": name, "start_date": day(first["START"]),
                         "stop_date": None if ongoing else day(group["STOP"].max()),
                         "status": "ongoing" if ongoing else "stopped",
                         "encounter": first["ENCOUNTER"]})

    alg = allergies[allergies["PATIENT"] == pid]

    # Every encounter that a fact comes from becomes a "report" with a readable id
    used = set(lab_encounters.index) | set(cond["ENCOUNTER"]) | {m["encounter"] for m in med_rows} | set(alg["ENCOUNTER"])
    enc = encounters[encounters["Id"].isin(used)].sort_values("START")
    org_names = dict(zip(organisations["Id"], organisations["NAME"]))
    report_id = {}
    reports = []
    for n, row in enumerate(enc.itertuples(), start=1):
        rid = f"{label}-R{n:02d}"
        report_id[row.Id] = rid
        has_labs = row.Id in lab_encounters.index
        reports.append({"report_id": rid, "report_date": day(row.START),
                        "report_type": "lab report" if has_labs else "visit record",
                        "description": row.DESCRIPTION, "facility": org_names.get(row.ORGANIZATION)})
    facility = {r["report_id"]: r["facility"] for r in reports}

    test_results = []
    for row in labs.sort_values("DATE").itertuples():
        name, unit, low, high = TEST_MAP[row.DESCRIPTION]
        if name == "Hemoglobin":
            low, high = HEMOGLOBIN_RANGE[sex]
        rid = report_id[row.ENCOUNTER]
        test_results.append({
            "report_id": rid, "patient_id": f"SYN-{label}", "test_name": name,
            "value": round(float(row.value), 2), "unit": unit, "ref_low": low, "ref_high": high,
            "test_date": row.day, "lab_name": facility[rid],
        })

    # Blood pressure on the lab-report days, and latest height/weight for the profile
    vitals = obs[obs["ENCOUNTER"].isin(lab_encounters.index)]
    bp = vitals[vitals["DESCRIPTION"].isin(["Systolic Blood Pressure", "Diastolic Blood Pressure"])]
    bp = bp.pivot_table(index=["ENCOUNTER", "day"], columns="DESCRIPTION", values="VALUE", aggfunc="first").reset_index()
    bp_readings = [{"systolic": round(float(r["Systolic Blood Pressure"])), "diastolic": round(float(r["Diastolic Blood Pressure"])),
                    "date": r["day"], "source": report_id[r["ENCOUNTER"]]}
                   for _, r in bp.sort_values("day").iterrows()]

    def latest_vital(description):
        v = obs[obs["DESCRIPTION"] == description].sort_values("DATE")
        return float(v.iloc[-1]["VALUE"]) if len(v) else None

    age = int((last_day - pd.Timestamp(person["BIRTHDATE"])).days // 365.25)
    profile = {
        "age": age, "sex": sex, "pregnant": False,
        "height_cm": latest_vital("Body Height"), "weight_kg": latest_vital("Body Weight"),
        "systolic_bp": bp_readings[-1]["systolic"] if bp_readings else None,
        "diastolic_bp": bp_readings[-1]["diastolic"] if bp_readings else None,
        "bp_date": bp_readings[-1]["date"] if bp_readings else None,
    }

    return {
        "format_version": "medtrail-history-0.1",
        "patient_id": f"SYN-{label}",
        "synthetic": True,
        "demo_case": f"Patient {label} ({focus})",
        "notice": "Synthetic demonstration record generated by Synthea. Not a real patient and not clinical ground truth.",
        "source": {"dataset": "Synthea 1K sample, CSV (nov2021)", "synthea_patient_id": pid},
        "reference_ranges_note": "Synthea gives no reference ranges; typical adult ranges were added for this demonstration.",
        "data_cleaning_notes": cleaning_notes,
        "profile": profile,
        "reports": reports,
        "test_results": test_results,
        "conditions": [{"name": r.DESCRIPTION, "onset_date": day(r.START), "end_date": day(r.STOP),
                        "status": "active" if pd.isna(r.STOP) else "resolved", "report_id": report_id[r.ENCOUNTER]}
                       for r in cond.itertuples()],
        "medications": [{**{k: v for k, v in m.items() if k != "encounter"}, "report_id": report_id[m["encounter"]]}
                        for m in med_rows],
        "allergies": [{"substance": r.DESCRIPTION, "reaction": r.DESCRIPTION1 if pd.notna(r.DESCRIPTION1) else None,
                       "recorded_date": day(r.START), "report_id": report_id[r.ENCOUNTER]}
                      for r in alg.itertuples()],
        "bp_readings": bp_readings,
    }


def main():
    print("Loading Synthea CSV files (observations.csv is large, this takes a few seconds)...")
    data = (
        load_csv("patients", ["Id", "BIRTHDATE", "GENDER"]),
        load_csv("encounters", ["Id", "START", "PATIENT", "ORGANIZATION", "DESCRIPTION"]),
        load_csv("observations", ["DATE", "PATIENT", "ENCOUNTER", "DESCRIPTION", "VALUE"]),
        load_csv("conditions", ["START", "STOP", "PATIENT", "ENCOUNTER", "DESCRIPTION"]),
        load_csv("medications", ["START", "STOP", "PATIENT", "ENCOUNTER", "DESCRIPTION"]),
        load_csv("allergies", ["START", "PATIENT", "ENCOUNTER", "DESCRIPTION", "DESCRIPTION1"]),
        load_csv("organizations", ["Id", "NAME"]),
    )
    for label, (prefix, focus) in DEMO_PATIENTS.items():
        history = build_patient(label, prefix, focus, data)
        path = OUT / f"patient_{label}_{focus}.json"
        path.write_text(json.dumps(history, indent=2))
        p = history["profile"]
        print(f"\nPatient {label} ({focus}): {p['age']}-year-old {p['sex']} -> {path.name}")
        print(f"  {len(history['reports'])} reports ({sum(r['report_type'] == 'lab report' for r in history['reports'])} lab), "
              f"{len(history['test_results'])} test results, {len(history['conditions'])} conditions, "
              f"{len(history['medications'])} medications, {len(history['allergies'])} allergies, "
              f"{len(history['bp_readings'])} BP readings")
        dates = [r["report_date"] for r in history["reports"] if r["report_type"] == "lab report"]
        print(f"  lab reports from {dates[0]} to {dates[-1]}")


if __name__ == "__main__":
    main()
