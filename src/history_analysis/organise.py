"""
Organise one patient's medical history and run every MedTrail analysis on it.

Input: a history in the MedTrail history format (docs/history_format.md).
Output: one dictionary with the history grouped by type plus the results of
trend analysis, rule-based screening and the ML risk models.
"""
import re

from src.prediction.predict import predict_all
from src.screening.screening import screen_all
from src.trends.trends import analyse_trends

# If one of these diagnoses is already recorded, the matching ML screening estimate is not
# meaningful (the models look for UNdiagnosed risk; treatment changes the input values).
DOCUMENTED_PATTERNS = {
    "diabetes": r"(?<!pre)diabetes",
    "kidney": r"kidney|renal",
    "liver": r"liver|hepat|cirrhosis",
    "heart": r"heart|coronary|myocard|angina",
}


def clean_name(name):
    """'Anemia (disorder)' -> 'Anemia'."""
    return re.sub(r"\s*\((?:disorder|finding|situation|procedure)\)?\s*$", "", str(name)).strip()


def analyse_history(history):
    reports = sorted(history.get("reports", []), key=lambda r: r["report_date"])
    conditions = [{**c, "name": clean_name(c["name"])} for c in history.get("conditions", [])]
    medications = history.get("medications", [])
    test_results = history.get("test_results", [])
    profile = history.get("profile", {})

    risks = predict_all(profile, test_results)
    active = [c for c in conditions if c["status"] == "active"]
    for disease, risk in risks.items():
        found = [c for c in active if re.search(DOCUMENTED_PATTERNS[disease], c["name"], re.IGNORECASE)]
        if found:
            risk["already_documented"] = [{"name": c["name"], "report_id": c["report_id"]} for c in found]

    return {
        "patient_id": history.get("patient_id"),
        "synthetic": history.get("synthetic", False),
        "notice": history.get("notice"),
        "profile": profile,
        "reports": reports,
        "conditions": {"active": active, "resolved": [c for c in conditions if c["status"] != "active"]},
        "medications": {"ongoing": [m for m in medications if m["status"] == "ongoing"],
                        "stopped": [m for m in medications if m["status"] != "ongoing"]},
        "allergies": history.get("allergies", []),
        "bp_readings": history.get("bp_readings", []),
        "trends": analyse_trends(test_results),
        "screening": screen_all(profile, test_results, history.get("bp_readings")),
        "risks": risks,
        "data_notes": [n for n in [history.get("reference_ranges_note"), *history.get("data_cleaning_notes", [])] if n],
    }
