# MedTrail – Member 5 Dataset Notes

Last updated: 24 November 2026. Every dataset below is public or synthetic. No real patient records are used anywhere in this module.

All numbers in this file come from running the project's own scripts:

- `python src/preprocessing/download_datasets.py` (disease datasets)
- `python src/history_analysis/explore_synthea.py` (Synthea)

---

## 1. Disease risk datasets

| Disease | Dataset and source | Licence | Size | Target as downloaded | Missing values as downloaded | Notes and limitations |
|---|---|---|---|---|---|---|
| Type 2 diabetes | Pima Indians Diabetes, [OpenML dataset 37](https://www.openml.org/d/37) (a copy of the original UCI file) | Public (OpenML) | 768 rows × 9 columns | `Outcome`: 0 = 500, 1 = 268 | None stored as blank, but zeros mean "not recorded": Glucose 5, BloodPressure 35, SkinThickness 227, Insulin 374, BMI 11 | Women of Pima heritage aged 21+. "Glucose" is a 2-hour glucose tolerance test, and BloodPressure is diastolic only. |
| Chronic kidney disease | [UCI dataset 336](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) | CC BY 4.0 | 400 rows × 25 columns | `class`: ckd 248 + `ckd\t` 2 (tab typo, really ckd) = 250; notckd 150 | 24 of 25 columns have gaps; worst are rbc 152, rbcc 131, wbcc 106 | From a hospital in Karaikudi, Tamil Nadu. Small dataset; very high accuracy is typical for it and is not proof of a strong model. |
| Liver disease | [UCI dataset 225 (ILPD)](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset) | CC BY 4.0 | 583 rows × 11 columns | `Selector`: 1 = liver patient 416, 2 = not 167 | A/G Ratio 4 | From north-east Andhra Pradesh. Imbalanced: always guessing "patient" scores 71% accuracy, so we report recall and ROC-AUC. |
| Heart disease | [UCI dataset 45 (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) | CC BY 4.0 | 303 rows × 14 columns | `num`: 0 = 164; 1–4 = 139 (55 / 36 / 35 / 13), converted to 0 = no disease, 1 = disease | ca 4, thal 2 (not used by our model) | US patients. MedTrail uses only 5 of the 13 inputs (age, sex, resting BP, cholesterol, fasting blood sugar), so this model is shown as indicative only. |
| Thyroid (stretch goal) | UCI thyroid0387 | – | – | – | – | Not downloaded. Only if time allows. |

### Source change for the diabetes dataset

The project introduction lists the Pima source as "Kaggle or UCI". On 24 November 2026 neither was available: the Kaggle copy (`uciml/pima-indians-diabetes-database`) showed "We can't find that page", and the UCI page returned "404 Dataset not found". We switched to OpenML dataset 37, which holds the same 768 records (500 negative, 268 positive).

If the teammate who built the original diabetes model shares their `diabetes.csv`, place it in `datasets/raw/diabetes/` and the download script will use it instead, so the results match the report.

---

## 2. Medical-history dataset: Synthea

| Field | Details |
|---|---|
| Name | Synthea synthetic patient records, "1K Sample Synthetic Patient Records, CSV" (Previous Versions 2019–2021) |
| Official source | Synthea by The MITRE Corporation – [synthetichealth.github.io/downloads.html](https://synthetichealth.github.io/downloads.html) |
| File | `synthea_sample_data_csv_nov2021.zip` (about 57 MB as downloaded), unzipped to `datasets/synthetic_patients/raw/csv/` |
| Access | Free direct download, no account needed |
| Format | 18 CSV files linked by patient ID |
| Licence | The downloads page states the data is free from cost, privacy and security restrictions and can be used without restriction. Synthea software is Apache-2.0. |
| Citation | Walonoski J, Kramer M, Nichols J, Quina A, Moesel C, Hall D, Duffett C, Dube K, Gallagher T, McLachlan S. Synthea: An approach, method, and software mechanism for generating synthetic patients and the synthetic electronic health care record. *Journal of the American Medical Informatics Association* 2018; 25(3): 230–238. https://doi.org/10.1093/jamia/ocx079 |

### What the sample contains

| File | Rows | What it gives the summaries |
|---|---|---|
| patients.csv | 1,163 | Profile (age, sex). Names are fake and are never sent to Gemini. |
| encounters.csv | 61,459 | Visits with dates, which form the timeline |
| conditions.csv | 38,094 | Recorded diagnoses |
| medications.csv | 56,430 (1,117 patients) | Medication history |
| allergies.csv | 794 (179 patients) | Allergy history |
| observations.csv | 531,144 | Lab results and vital signs with dates |

### Lab tests relevant to MedTrail

Trend analysis needs at least 3 results for the same test.

| Synthea test name | MedTrail use | Unit | Patients | Patients with 3+ results |
|---|---|---|---|---|
| Hemoglobin A1c/Hemoglobin.total in Blood | HbA1c – Patient A, trends | % | 382 | 355 |
| Glucose | Diabetes input, trends | mg/dL | 437 | 402 |
| Creatinine | Serum creatinine – CKD input, Patient B | mg/dL | 437 | 402 |
| Urea Nitrogen | Blood urea – CKD input | mg/dL | 437 | 402 |
| Glomerular filtration rate/1.73 sq M.predicted | eGFR – Patient B context | mL/min | 207 | 184 |
| Microalbumin Creatinine Ratio | Urine albumin – Patient B | mg/g | 71 | 67 |
| Hemoglobin [Mass/volume] in Blood | Haemoglobin – anaemia screening, CKD input | g/dL | 1,160 | 165 |
| Alanine aminotransferase | SGPT – Patient C, liver input | U/L | 201 | 184 |
| Aspartate aminotransferase | SGOT – Patient C, liver input | U/L | 201 | 184 |
| Bilirubin.total in Serum or Plasma | Total bilirubin – Patient C, liver input | mg/dL | 201 | 184 |
| Albumin in Serum or Plasma | Albumin – liver input | g/dL | 201 | 184 |
| Systolic / Diastolic Blood Pressure | BP screening, model inputs | mm[Hg] | 1,158 | 1,155 |
| Total Cholesterol | Heart input | mg/dL | 721 | 605 |

### Recorded diagnoses (patients)

Prediabetes 341 · Diabetes 73 · neuropathy due to type 2 diabetes 24 · diabetic retinopathy 21 and 13 · chronic kidney disease stage 1: 30, stage 2: 1 · anaemia 324 · hypertension 292 · liver disease: none found.

### Demonstration patient plan (finalised on Day 5)

- **Patient A (diabetes):** a patient with a diabetes diagnosis, 3+ HbA1c and glucose results, and medication records.
- **Patient B (kidney):** a patient with chronic kidney disease and repeated creatinine, eGFR, urine albumin ratio and haemoglobin results.
- **Patient C (liver):** liver tests exist for 184 patients with 3+ results, but no patient has a liver diagnosis. We pick the patient whose ALT and bilirubin change the most. If no values leave the normal range, we either show stable results honestly or adjust values and label the case "modified synthetic case".

All three will be clearly labelled synthetic demonstration cases, not real patients or clinical ground truth.

### Notes and limitations

- Synthea test names are long standard (LOINC-style) names. They must be mapped to MedTrail's standard test names from Member 4 (for example, Alanine aminotransferase → SGPT).
- Synthea's glucose is not labelled fasting, random or post-meal, so feeding it into the diabetes model's glucose input is an approximation.
- The population is synthetic and American (Massachusetts by default), not Indian.
- Synthetic records are complete and tidy. Real records have gaps and errors, so results on Synthea will look cleaner than real use.
- Lab values are produced by simulation models, not by real laboratories.

---

## 3. Datasets for evaluating the summaries

| Field | Synthea | MTSamples | MTS-Dialog | PhysioNet (MIMIC-based: patient summaries, MIMIC-IV-Ext-BHC, Discharge Me, MeDiSumQA) |
|---|---|---|---|---|
| Official source | synthetichealth.github.io | [Kaggle: tboyle10/medicaltranscriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions); backup copy on Hugging Face: argilla/medical-domain | [github.com/abachaa/MTS-Dialog](https://github.com/abachaa/MTS-Dialog) | physionet.org |
| Access | Free download | Free (Kaggle login) | Free, public repository | **Credentialed**: account, training and a data use agreement |
| Format | CSV | CSV | CSV | CSV / text |
| Records | 1,163 patients | 4,999 notes | 1,201 train, 100 validation, 2 × 200 test | Thousands |
| Multi-visit patient history | Yes | No (single notes) | No (short conversations) | Yes |
| Reference summaries | No | Only a one-line `description` | Yes: conversation → clinical-note section | Yes |
| Patient-friendly references | No | No | No | Yes (discharge instructions) |
| Doctor-friendly references | No | Partial (one line) | Partial (note sections) | Yes (hospital-course summaries) |
| Licence | Use without restriction | Listed as CC0 on Kaggle (scraped from mtsamples.com) | CC BY 4.0 (cite the EACL 2023 paper) | PhysioNet credentialed data use agreement |
| Limitations | No reference summaries | Not patient histories; weak references | Snippets, not histories | Not obtainable within 7 days |
| Decision | **Input histories for the demo** | Optional sanity check | Optional ROUGE / BERTScore check | Cited as ideal but restricted |

**Conclusion:** no openly accessible dataset pairs multi-visit patient histories with both patient-friendly and doctor-friendly reference summaries. MedTrail therefore:

1. uses Synthea for the input histories;
2. evaluates its summaries with automatic grounding checks (every fact and number must trace back to a source report) plus a human-review checklist;
3. optionally runs ROUGE or BERTScore on MTS-Dialog as a sanity check of the language model's summarising ability. A high score is not proof of medical correctness.

Fine-tuning a language model is not planned: there is no openly accessible paired data and not enough time, and prompting a pretrained model is sufficient.

The MTSamples Kaggle page has not been re-checked since the Pima copy disappeared from Kaggle. Verify it before use.

---

## 4. Privacy and usage rules

- Only public or synthetic data is used.
- Gemini receives only de-identified synthetic test values, never names or addresses, because free-tier content may be used by Google to improve its products.
- Large raw Synthea files are kept out of Git by `.gitignore`. Only the three selected demonstration patients will be committed.
- Every result is a screening estimate from a small public dataset, never a diagnosis.
