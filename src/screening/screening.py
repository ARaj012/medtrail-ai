"""
Rule-based screening flags (from the project document, based on WHO cut-offs).

Anaemia:  haemoglobin below 13 g/dL (men) or 12 g/dL (non-pregnant women), aged 15+.
          Severity: mild = 11.0 up to the cut-off, moderate = 8.0-10.9, severe = below 8.0.
High BP:  systolic >= 140 and/or diastolic >= 90 mmHg on two different days (adults 18+).
Pregnant women and children have different cut-offs, so both checks return "not_applicable".

These are screening flags, not diagnoses.
"""
from collections import defaultdict

DISCLAIMER = "Screening flag based on WHO cut-offs. Not a diagnosis; a doctor must confirm."

ANAEMIA_CUTOFF = {"male": 13.0, "female": 12.0}      # g/dL, non-pregnant adults (15+)
ANAEMIA_MIN_AGE = 15
BP_MIN_AGE = 18
SYSTOLIC_LIMIT, DIASTOLIC_LIMIT = 140, 90
HEMOGLOBIN_NAMES = {"hemoglobin", "haemoglobin", "hb", "hgb"}


def _sex(profile):
    return {"male": "male", "m": "male", "female": "female", "f": "female"}.get(
        str(profile.get("sex", "")).strip().lower())


def _not_applicable_reason(profile, min_age):
    if profile.get("pregnant"):
        return "Not applicable during pregnancy (different cut-offs)."
    age = profile.get("age")
    if age is not None and age < min_age:
        return f"Not applicable under age {min_age} (children have different cut-offs)."
    return None


def latest_hemoglobin(test_results):
    """Most recent haemoglobin result, converted to g/dL if the report used g/L."""
    found = [r for r in test_results
             if str(r.get("test_name", "")).strip().lower() in HEMOGLOBIN_NAMES and r.get("value") is not None]
    if not found:
        return None
    result = max(found, key=lambda r: str(r.get("test_date", "")))
    value = float(result["value"])
    if str(result.get("unit", "")).replace(" ", "").lower() == "g/l":
        value = value / 10
    return {"value": round(value, 1), "unit": "g/dL", "test_date": result.get("test_date"),
            "report_id": result.get("report_id")}


def screen_anaemia(profile, test_results):
    base = {"check": "anaemia", "disclaimer": DISCLAIMER}

    reason = _not_applicable_reason(profile, ANAEMIA_MIN_AGE)
    if reason:
        return {**base, "status": "not_applicable", "reason": reason}

    sex = _sex(profile)
    if sex is None or profile.get("age") is None:
        return {**base, "status": "needs_more_data", "reason": "Needs the patient's sex and age."}

    hb = latest_hemoglobin(test_results)
    if hb is None:
        return {**base, "status": "needs_more_data", "reason": "Needs a Hemoglobin result."}

    cutoff = ANAEMIA_CUTOFF[sex]
    value = hb["value"]
    if value < 8.0:
        severity = "severe"
    elif value < 11.0:
        severity = "moderate"
    elif value < cutoff:
        severity = "mild"
    else:
        severity = None

    who = "men" if sex == "male" else "non-pregnant women"
    rule = f"WHO: anaemia if haemoglobin is below {cutoff:g} g/dL for {who} aged 15+"
    if severity:
        return {**base, "status": "flag", "severity": severity, "hemoglobin": hb, "cutoff": cutoff,
                "rule": rule,
                "reason": f"Haemoglobin {value:g} g/dL is below {cutoff:g} g/dL ({severity})."}
    return {**base, "status": "normal", "severity": None, "hemoglobin": hb, "cutoff": cutoff,
            "rule": rule, "reason": f"Haemoglobin {value:g} g/dL is at or above {cutoff:g} g/dL."}


def daily_bp(bp_readings):
    """Average the readings taken on the same day; return days from newest to oldest."""
    by_day = defaultdict(list)
    for reading in bp_readings:
        if reading.get("systolic") is None or reading.get("diastolic") is None:
            continue
        by_day[str(reading.get("date"))].append(reading)
    days = []
    for day, readings in by_day.items():
        systolic = sum(r["systolic"] for r in readings) / len(readings)
        diastolic = sum(r["diastolic"] for r in readings) / len(readings)
        days.append({
            "date": day,
            "systolic": round(systolic), "diastolic": round(diastolic),
            "raised": systolic >= SYSTOLIC_LIMIT or diastolic >= DIASTOLIC_LIMIT,
            "sources": sorted({str(r.get("source")) for r in readings if r.get("source")}),
        })
    return sorted(days, key=lambda d: d["date"], reverse=True)


def screen_blood_pressure(profile, bp_readings=None):
    """Flag high BP only when the two most recent days with readings are both raised."""
    base = {"check": "high_blood_pressure", "disclaimer": DISCLAIMER,
            "rule": "WHO: systolic >= 140 and/or diastolic >= 90 mmHg on two different days"}

    reason = _not_applicable_reason(profile, BP_MIN_AGE)
    if reason:
        return {**base, "status": "not_applicable", "reason": reason}

    readings = list(bp_readings or [])
    if not readings and profile.get("systolic_bp") is not None and profile.get("diastolic_bp") is not None:
        readings = [{"systolic": profile["systolic_bp"], "diastolic": profile["diastolic_bp"],
                     "date": profile.get("bp_date", "latest profile reading"), "source": "profile"}]

    days = daily_bp(readings)
    if not days:
        return {**base, "status": "needs_more_data", "reason": "Needs at least one blood-pressure reading."}

    recent = days[:2]
    shown = [f"{d['systolic']}/{d['diastolic']} on {d['date']}" for d in recent]
    if len(recent) == 2 and all(d["raised"] for d in recent):
        return {**base, "status": "flag", "readings_used": recent,
                "reason": "Raised on two different days: " + " and ".join(shown) + "."}
    if recent[0]["raised"]:
        return {**base, "status": "raised_once", "readings_used": recent,
                "reason": f"Latest reading {shown[0]} is raised. Repeat on another day to confirm."}
    return {**base, "status": "normal", "readings_used": recent,
            "reason": f"Latest reading {shown[0]} is below 140/90."}


def screen_all(profile, test_results, bp_readings=None):
    return {"anaemia": screen_anaemia(profile, test_results),
            "high_blood_pressure": screen_blood_pressure(profile, bp_readings)}


if __name__ == "__main__":
    import json
    demo_profile = {"age": 45, "sex": "female", "pregnant": False}
    demo_results = [
        {"report_id": "R201", "test_name": "Hemoglobin", "value": 12.4, "unit": "g/dL", "test_date": "2026-05-02"},
        {"report_id": "R205", "test_name": "Hemoglobin", "value": 10.2, "unit": "g/dL", "test_date": "2026-10-18"},
    ]
    demo_bp = [
        {"systolic": 146, "diastolic": 92, "date": "2026-10-18", "source": "R205"},
        {"systolic": 150, "diastolic": 88, "date": "2026-11-02", "source": "R207"},
    ]
    print(json.dumps(screen_all(demo_profile, demo_results, demo_bp), indent=2))