"""
Trend analysis across a patient's reports.

For every test: take the latest 3 results (oldest -> newest), check the direction of change,
compare the latest value with its reference range, and decide whether to raise an alert.

A change is only called "worsening" when it moves AWAY from the reference range
(e.g. haemoglobin rising towards normal is improving, not harmful).

Alert severity:
  high      out of range AND worsening in each of the last 3 reports,
            or more than 50% beyond the range limit
  moderate  out of range (not worsening steadily), or close to a limit AND worsening
  low       close to a limit (within 10% of the range width)
  (none)    inside the range, or no reference range to judge against
"""

DISCLAIMER = "Trend alerts are monitoring aids based on reference ranges. Not a diagnosis."
MIN_CHANGE = 0.05     # a trend must change by at least 5% overall, so tiny lab noise is ignored
NEAR_LIMIT = 0.10     # "close to a limit" = within 10% of the reference range width
FAR_OUTSIDE = 0.50    # more than 50% beyond a limit counts as high severity
SEVERITY_ORDER = {"high": 0, "moderate": 1, "low": 2}


def _unit(value):
    return str(value or "").replace(" ", "").lower()


def _group_by_test(test_results):
    groups = {}
    for result in test_results:
        if result.get("value") is None or not result.get("test_name"):
            continue
        groups.setdefault(str(result["test_name"]).strip(), []).append(result)
    return groups


def _direction(values):
    """rising / falling / stable / mixed for the last 3 values; needs 3 values."""
    if len(values) < 3:
        return "not_enough_data"
    steps = [b - a for a, b in zip(values, values[1:])]
    base = abs(values[0]) or 1.0
    if abs(values[-1] - values[0]) / base < MIN_CHANGE:
        return "stable"
    if all(step > 0 for step in steps):
        return "rising"
    if all(step < 0 for step in steps):
        return "falling"
    return "mixed"


def _range_status(value, low, high):
    if low is None and high is None:
        return "no_reference_range"
    if high is not None and value > high:
        return "high"
    if low is not None and value < low:
        return "low"
    width = (high - low) if (low is not None and high is not None) else abs(high if high is not None else low)
    margin = NEAR_LIMIT * width
    if high is not None and value >= high - margin:
        return "near_high"
    if low is not None and value <= low + margin:
        return "near_low"
    return "normal"


def _meaning(direction, status):
    """Is the change moving away from the reference range (worsening) or back towards it?"""
    if direction in ("not_enough_data", "stable", "mixed"):
        return direction
    if status == "no_reference_range":
        return "unclear"
    going_up = direction == "rising"
    if status in ("high", "near_high"):
        return "worsening" if going_up else "improving"
    if status in ("low", "near_low"):
        return "improving" if going_up else "worsening"
    return "within_range"


def _far_outside(value, low, high):
    if high is not None and value > high and high > 0:
        return (value - high) / high > FAR_OUTSIDE
    if low is not None and value < low and low > 0:
        return (low - value) / low > FAR_OUTSIDE
    return False


def _severity(status, meaning, far):
    if status in ("high", "low"):
        return "high" if (meaning == "worsening" or far) else "moderate"
    if status in ("near_high", "near_low"):
        return "moderate" if meaning == "worsening" else "low"
    return None


def _explain(name, unit, points, status, direction, meaning, low, high):
    series = " -> ".join(f"{p['value']:g}" for p in points)
    unit_text = f" {unit}" if unit else ""
    if low is not None and high is not None:
        range_text = f"reference range {low:g}-{high:g}{unit_text}"
    elif high is not None:
        range_text = f"upper limit {high:g}{unit_text}"
    elif low is not None:
        range_text = f"lower limit {low:g}{unit_text}"
    else:
        range_text = "no reference range given"

    trend_text = {
        "rising": f"rose in each of the last {len(points)} reports ({series}{unit_text})",
        "falling": f"fell in each of the last {len(points)} reports ({series}{unit_text})",
        "stable": f"stayed about the same ({series}{unit_text})",
        "mixed": f"went up and down ({series}{unit_text})",
        "not_enough_data": f"has {len(points)} result(s) so far ({series}{unit_text})",
    }[direction]
    status_text = {
        "high": "is now above the", "low": "is now below the",
        "near_high": "is now close to the upper end of the", "near_low": "is now close to the lower end of the",
        "normal": "is within the", "no_reference_range": "has",
    }[status]
    sentence = f"{name} {trend_text} and {status_text} {range_text}."
    if meaning == "improving":
        sentence += " The change is towards the reference range (improving)."
    if status == "no_reference_range":
        sentence += " Without a range, MedTrail does not judge whether the change is good or bad."
    return sentence


def analyse_test(name, results):
    """Trend and alert for one test."""
    results = sorted(results, key=lambda r: str(r.get("test_date", "")))
    latest = results[-1]
    same_unit = [r for r in results if _unit(r.get("unit")) == _unit(latest.get("unit"))]
    skipped_units = len(results) - len(same_unit)
    recent = same_unit[-3:]

    points = [{"value": float(r["value"]), "test_date": r.get("test_date"), "report_id": r.get("report_id")}
              for r in recent]
    values = [p["value"] for p in points]
    low, high = latest.get("ref_low"), latest.get("ref_high")
    low = float(low) if low is not None else None
    high = float(high) if high is not None else None

    status = _range_status(values[-1], low, high)
    direction = _direction(values)
    meaning = _meaning(direction, status)
    severity = _severity(status, meaning, _far_outside(values[-1], low, high))

    result = {
        "test_name": name,
        "unit": latest.get("unit"),
        "latest": points[-1],
        "previous": points[:-1],
        "reference_range": {"low": low, "high": high},
        "range_status": status,
        "direction": direction,
        "trend_meaning": meaning,
        "change": round(values[-1] - values[0], 3) if len(values) > 1 else None,
        "alert": severity is not None,
        "severity": severity,
        "explanation": _explain(name, latest.get("unit"), points, status, direction, meaning, low, high),
        "source_report_ids": [p["report_id"] for p in points if p.get("report_id")],
    }
    if skipped_units:
        result["note"] = f"{skipped_units} earlier result(s) used a different unit and were not compared."
    return result


def analyse_trends(test_results):
    """Trend analysis for every test; alerts sorted with the most serious first."""
    analyses = [analyse_test(name, results) for name, results in _group_by_test(test_results).items()]
    alerts = sorted((a for a in analyses if a["alert"]), key=lambda a: SEVERITY_ORDER[a["severity"]])
    return {"tests_analysed": len(analyses), "alert_count": len(alerts), "alerts": alerts,
            "all_tests": sorted(analyses, key=lambda a: a["test_name"]), "disclaimer": DISCLAIMER}


if __name__ == "__main__":
    import json

    def r(rid, name, value, unit, low, high, date):
        return {"report_id": rid, "patient_id": "P001", "test_name": name, "value": value, "unit": unit,
                "ref_low": low, "ref_high": high, "test_date": date, "lab_name": "Demo Lab"}

    demo = [
        r("R101", "HbA1c", 5.9, "%", 4.0, 5.6, "2026-04-10"),
        r("R104", "HbA1c", 6.4, "%", 4.0, 5.6, "2026-07-12"),
        r("R108", "HbA1c", 6.8, "%", 4.0, 5.6, "2026-10-20"),
        r("R101", "Hemoglobin", 10.1, "g/dL", 12.0, 15.5, "2026-04-10"),
        r("R104", "Hemoglobin", 10.9, "g/dL", 12.0, 15.5, "2026-07-12"),
        r("R108", "Hemoglobin", 11.6, "g/dL", 12.0, 15.5, "2026-10-20"),
        r("R101", "Serum Creatinine", 0.9, "mg/dL", 0.6, 1.2, "2026-04-10"),
        r("R108", "Serum Creatinine", 0.92, "mg/dL", 0.6, 1.2, "2026-10-20"),
    ]
    out = analyse_trends(demo)
    print(f"{out['tests_analysed']} tests analysed, {out['alert_count']} alerts\n")
    for a in out["all_tests"]:
        print(f"[{a['severity'] or 'no alert'}] {a['explanation']}")