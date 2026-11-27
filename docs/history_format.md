# MedTrail – Medical-history input format (proposal, version 0.1)

Proposed by Member 5 (ML and AI summaries) for review by Member 2 (backend) and Member 4 (OCR).

## Why this is needed

The agreed data contract describes **one test result**. The patient and doctor summaries need a
patient's **whole history**: many reports, plus diagnoses, medicines and allergies. This proposal
**adds to** the contract and does not change it: every item in `test_results` is exactly the
agreed test-result format.

## The format

One JSON object per patient:

| Field | Type | Meaning |
|---|---|---|
| `format_version` | text | `"medtrail-history-0.1"` |
| `patient_id` | text | The patient's MedTrail id (synthetic ids start with `SYN-`) |
| `synthetic` | true/false | `true` for demonstration records |
| `profile` | object | `age`, `sex`, `pregnant`, `height_cm`, `weight_kg`, `systolic_bp`, `diastolic_bp`, `bp_date` |
| `reports` | list | One entry per source document: `report_id`, `report_date`, `report_type` (`"lab report"` / `"visit record"`), `description`, `facility` |
| `test_results` | list | **Unchanged team data contract**: `report_id`, `patient_id`, `test_name`, `value`, `unit`, `ref_low`, `ref_high`, `test_date`, `lab_name` |
| `conditions` | list | `name`, `onset_date`, `end_date`, `status` (`"active"` / `"resolved"`), `report_id` |
| `medications` | list | `name`, `start_date`, `stop_date`, `status` (`"ongoing"` / `"stopped"`), `report_id` |
| `allergies` | list | `substance`, `reaction`, `recorded_date`, `report_id` (an empty list means none recorded) |
| `bp_readings` | list | `systolic`, `diastolic`, `date`, `source` (the report id) |

**Rule:** every condition, medicine, allergy, test result and BP reading carries the `report_id`
of the document it came from. The summaries cite these ids, and the grounding check uses them to
reject any statement that has no source.

## Example (shortened, from synthetic Patient A)

```json
{
  "format_version": "medtrail-history-0.1",
  "patient_id": "SYN-A",
  "synthetic": true,
  "profile": {"age": 51, "sex": "male", "pregnant": false, "height_cm": 173.1, "weight_kg": 101.4,
              "systolic_bp": 145, "diastolic_bp": 101, "bp_date": "2021-03-06"},
  "reports": [
    {"report_id": "A-R10", "report_date": "2021-03-06", "report_type": "lab report",
     "description": "General examination of patient (procedure)", "facility": "TRISTAN MEDICAL PC"}
  ],
  "test_results": [
    {"report_id": "A-R10", "patient_id": "SYN-A", "test_name": "HbA1c", "value": 7.0, "unit": "%",
     "ref_low": 4.0, "ref_high": 5.6, "test_date": "2021-03-06", "lab_name": "TRISTAN MEDICAL PC"}
  ],
  "conditions": [
    {"name": "Diabetes", "onset_date": "2010-01-02", "end_date": null, "status": "active", "report_id": "A-R02"}
  ],
  "medications": [
    {"name": "24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet",
     "start_date": "2010-01-02", "stop_date": null, "status": "ongoing", "report_id": "A-R02"}
  ],
  "allergies": [],
  "bp_readings": [{"systolic": 145, "diastolic": 101, "date": "2021-03-06", "source": "A-R10"}]
}
```

## What each team member would do

- **Member 4 (OCR):** nothing new for lab reports; keep producing the agreed test-result format.
  Use the standard test names (for example `Serum Creatinine`, `SGPT`, `Hemoglobin`).
- **Member 2 (backend):** assemble this object from the `patients`, `reports` and `test_results`
  tables when calling `/ai/summary` and `/analyze`. Conditions, medicines and allergies come from
  the patient profile and any visit records.
- **Member 5 (ML/AI):** reads this object, analyses it and returns both summaries with source ids.

## Privacy

Names, phone numbers and addresses are **not** part of this format. Only these de-identified
values are sent to Gemini, because free-tier requests may be used by Google.

## The three synthetic demonstration patients

Built from the Synthea 1K CSV sample (nov2021) by `python -m src.history_analysis.build_history`:

| File | Case | Key history |
|---|---|---|
| `patient_A_diabetes.json` | 51-year-old man | Diabetes and hypertension; HbA1c 6.5 → 6.8 → 7.0 %; metformin, amlodipine, hydrochlorothiazide |
| `patient_B_kidney.json` | 57-year-old woman | Diabetic kidney disease; creatinine 1.7 → 2.2 → 2.3 mg/dL, eGFR 43.5 → 34.5 → 32.8; insulin, lisinopril |
| `patient_C_liver.json` | 71-year-old man | Diabetes and hyperlipidaemia; SGPT 24.4 → 34.7 → 59.6 U/L (now above range); simvastatin |

Notes on the demonstration data:

- These are synthetic records, not real patients or clinical ground truth.
- Synthea gives no reference ranges, so typical adult ranges were added for the demo.
- Synthea stores most Total Protein values in g/L under a g/dL label; the builder converts them
  and records this in `data_cleaning_notes`.
- None of the three patients has an allergy recorded. The summaries must say so, not invent one.
