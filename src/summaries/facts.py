"""
Build the fact sheet: a numbered list of facts taken ONLY from the analysed history.

Every summary (template or AI) is written from these facts, and every fact keeps
the report ids it came from. Each fact has a kind, so the doctor summary can keep
documented history, lab observations, ML estimates and missing information apart:

  profile | source_report | documented | medication | allergy | lab_observation
  trend | screening | ml_estimate | missing | data_note
"""


def num(value):
    """Short number text: 7.0 -> '7', 0.255 -> '0.255'."""
    return f"{value:g}"


def range_text(low, high, unit):
    unit = f" {unit}" if unit else ""
    if low is not None and high is not None:
        return f"{num(low)}-{num(high)}{unit}"
    if high is not None:
        return f"below {num(high)}{unit}"
    if low is not None:
        return f"above {num(low)}{unit}"
    return "no reference range"


STATUS_TEXT = {"high": "above the reference range", "low": "below the reference range",
               "near_high": "close to the upper limit", "near_low": "close to the lower limit",
               "normal": "within the reference range", "no_reference_range": "no reference range given"}


class FactSheet:
    def __init__(self):
        self.facts = []

    def add(self, kind, text, report_ids=(), numbers=(), **extra):
        fact = {"id": f"F{len(self.facts) + 1:02d}", "kind": kind, "text": text,
                "report_ids": sorted({r for r in report_ids if r}), "numbers": [float(n) for n in numbers], **extra}
        self.facts.append(fact)
        return fact["id"]


def build_facts(analysis):
    sheet = FactSheet()
    profile, reports = analysis["profile"], analysis["reports"]
    labs = [r for r in reports if r["report_type"] == "lab report"]

    # Profile and the source reports
    sheet.add("profile",
              f"{analysis['patient_id']} is a {profile.get('age')}-year-old {profile.get('sex')}. "
              f"The records contain {len(reports)} source reports ({len(labs)} lab reports) "
              f"dated {reports[0]['report_date']} to {reports[-1]['report_date']}.",
              [reports[0]["report_id"], reports[-1]["report_id"]],
              [profile.get("age"), len(reports), len(labs)], section="overview")
    for r in reports:
        sheet.add("source_report", f"{r['report_id']}: {r['report_type']} dated {r['report_date']} "
                  f"({r['description']}, {r['facility']}).", [r["report_id"]], section="sources",
                  report_id=r["report_id"], date=r["report_date"])

    # Documented history
    for c in analysis["conditions"]["active"]:
        sheet.add("documented", f"Documented diagnosis: {c['name']} (active, recorded since {c['onset_date']}).",
                  [c["report_id"]], section="conditions_active", name=c["name"])
    for c in analysis["conditions"]["resolved"]:
        sheet.add("documented", f"Previous diagnosis: {c['name']} (recorded {c['onset_date']}, resolved {c['end_date']}).",
                  [c["report_id"]], section="conditions_resolved", name=c["name"])
    for m in analysis["medications"]["ongoing"]:
        sheet.add("medication", f"Medication: {m['name']}, ongoing since {m['start_date']}.",
                  [m["report_id"]], section="medications_ongoing", name=m["name"])
    for m in analysis["medications"]["stopped"]:
        sheet.add("medication", f"Medication: {m['name']}, taken from {m['start_date']} to {m['stop_date']} (stopped).",
                  [m["report_id"]], section="medications_stopped", name=m["name"])
    if analysis["allergies"]:
        for a in analysis["allergies"]:
            reaction = f" (reaction: {a['reaction']})" if a.get("reaction") else ""
            sheet.add("allergy", f"Allergy recorded: {a['substance']}{reaction}, recorded {a['recorded_date']}.",
                      [a["report_id"]], section="allergies", name=a["substance"])
    else:
        sheet.add("missing", "No allergies are recorded in the supplied records.", section="allergies")

    # Laboratory observations and trends
    for t in analysis["trends"]["all_tests"]:
        latest, rng = t["latest"], t["reference_range"]
        lab_id = sheet.add(
            "lab_observation",
            f"Latest {t['test_name']}: {num(latest['value'])} {t['unit']} on {latest['test_date']} "
            f"(reference {range_text(rng['low'], rng['high'], t['unit'])}), {STATUS_TEXT[t['range_status']]}.",
            [latest["report_id"]], [latest["value"], *(v for v in (rng["low"], rng["high"]) if v is not None)],
            section="labs", test_name=t["test_name"], range_status=t["range_status"],
            alert=t["alert"], severity=t["severity"])
        trend_id = None
        if t["previous"]:
            points = t["previous"] + [latest]
            series = " -> ".join(num(p["value"]) for p in points)
            sheet_text = (f"{t['test_name']} over the last {len(points)} reports: {series} {t['unit']} "
                          f"({', '.join(p['test_date'] for p in points)}); direction: {t['direction']}"
                          f"{'' if t['trend_meaning'] in (t['direction'], 'within_range') else ', ' + t['trend_meaning']}.")
            trend_id = sheet.add("trend", sheet_text, [p["report_id"] for p in points],
                                 [p["value"] for p in points], section="trends", test_name=t["test_name"],
                                 direction=t["direction"], meaning=t["trend_meaning"],
                                 alert=t["alert"], severity=t["severity"])
        t["_fact_ids"] = [lab_id] + ([trend_id] if trend_id else [])

    # Rule-based screening (WHO cut-offs)
    bp = analysis["screening"]["high_blood_pressure"]
    readings = bp.get("readings_used", [])
    sheet.add("screening", f"Blood-pressure screening ({bp['rule']}): {bp['status']}. {bp['reason']}",
              [s for r in readings for s in r.get("sources", [])],
              [v for r in readings for v in (r["systolic"], r["diastolic"])], section="screening", check="bp",
              status=bp["status"])
    an = analysis["screening"]["anaemia"]
    hb = an.get("hemoglobin") or {}
    sheet.add("screening", f"Anaemia screening: {an['status']}. {an['reason']}",
              [hb.get("report_id")], [v for v in (hb.get("value"), an.get("cutoff")) if v is not None],
              section="screening", check="anaemia", status=an["status"])

    # ML screening estimates (clearly separated from documented facts)
    for disease, risk in analysis["risks"].items():
        if risk["status"] == "ok":
            text = (f"ML screening estimate (not a diagnosis) for {risk['label'].lower()}: {risk['risk_level']} risk "
                    f"(probability {num(risk['probability'])}, {risk['model']}). SHAP: {risk['explanation'].get('summary', '')} "
                    f"Model note: {risk['note']}")
            if risk.get("already_documented"):
                names = ", ".join(d["name"] for d in risk["already_documented"])
                text += (f" {names} is already a documented diagnosis, so this estimate of undiagnosed risk "
                         f"is not meaningful for this patient.")
            sheet.add("ml_estimate", text,
                      risk.get("source_report_ids", []) + [d["report_id"] for d in risk.get("already_documented", [])],
                      [risk["probability"]], section="ml", disease=disease, risk_level=risk["risk_level"],
                      already_documented=bool(risk.get("already_documented")))
        else:
            sheet.add("missing", f"ML screening estimate for {risk['label'].lower()} not calculated: {risk['message']}",
                      section="ml", disease=disease)

    for note in analysis["data_notes"]:
        sheet.add("data_note", note, section="data_notes")
    return sheet.facts


def fact_sheet_text(facts):
    """Facts as numbered lines, e.g. for printing or for an AI prompt."""
    return "\n".join(f"{f['id']} [{f['kind']}] {f['text']}"
                     + (f" Sources: {', '.join(f['report_ids'])}." if f["report_ids"] else "") for f in facts)
