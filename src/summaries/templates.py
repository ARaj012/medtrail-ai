"""
Template summaries: written ONLY from the fact sheet, with fixed sentence patterns.

They need no internet, cannot invent information, and are the fallback whenever the
AI summary is unavailable or fails the grounding check. Every sentence keeps the ids
of the facts it uses, so it can be traced back to source reports.

Run a demo from the project folder:
    python -m src.summaries.templates
"""
from datetime import date

from src.summaries.facts import build_facts, num, range_text

PLAIN = {
    "en": {
        "HbA1c": "HbA1c (average blood sugar over the last few months)",
        "Glucose": "blood sugar (glucose)",
        "Serum Creatinine": "creatinine (a kidney test)",
        "eGFR": "eGFR (how well the kidneys filter blood)",
        "Blood Urea Nitrogen": "blood urea nitrogen (a kidney test)",
        "Urine Albumin/Creatinine Ratio": "urine albumin (protein in the urine)",
        "Hemoglobin": "haemoglobin",
        "SGPT": "SGPT (a liver enzyme)", "SGOT": "SGOT (a liver enzyme)",
        "Total Bilirubin": "bilirubin (a liver test)",
        "Alkaline Phosphatase": "alkaline phosphatase (a liver test)",
        "Albumin": "albumin (a blood protein)", "Total Protein": "total protein",
        "Total Cholesterol": "total cholesterol",
        "LDL Cholesterol": "LDL cholesterol (often called 'bad' cholesterol)",
        "HDL Cholesterol": "HDL cholesterol (often called 'good' cholesterol)",
        "Triglycerides": "triglycerides (a type of blood fat)",
    },
    "hi": {
        "HbA1c": "HbA1c (पिछले कुछ महीनों की औसत ब्लड शुगर)",
        "Glucose": "ब्लड शुगर (ग्लूकोज़)",
        "Serum Creatinine": "क्रिएटिनिन (किडनी की जांच)",
        "eGFR": "eGFR (किडनी खून को कितनी अच्छी तरह साफ़ करती है)",
        "Blood Urea Nitrogen": "ब्लड यूरिया नाइट्रोजन (किडनी की जांच)",
        "Urine Albumin/Creatinine Ratio": "पेशाब में एल्ब्यूमिन (प्रोटीन)",
        "Hemoglobin": "हीमोग्लोबिन",
        "SGPT": "SGPT (लिवर एंज़ाइम)", "SGOT": "SGOT (लिवर एंज़ाइम)",
        "Total Bilirubin": "बिलीरुबिन (लिवर की जांच)",
        "Alkaline Phosphatase": "एल्कलाइन फॉस्फेटेज़ (लिवर की जांच)",
        "Albumin": "एल्ब्यूमिन (खून का प्रोटीन)", "Total Protein": "टोटल प्रोटीन",
        "Total Cholesterol": "टोटल कोलेस्ट्रॉल",
        "LDL Cholesterol": "LDL कोलेस्ट्रॉल ('खराब' कोलेस्ट्रॉल)",
        "HDL Cholesterol": "HDL कोलेस्ट्रॉल ('अच्छा' कोलेस्ट्रॉल)",
        "Triglycerides": "ट्राइग्लिसराइड (खून में एक तरह की चर्बी)",
    },
}

TEXT = {
    "en": {
        "titles": ["About your records", "Health conditions in your records", "Test results to look at",
                   "Blood pressure and haemoglobin", "Your medicines and allergies", "Points to discuss with your doctor"],
        "overview": "Your records include {n} reports dated {first} to {last}, including {k} lab reports.",
        "conditions": "Your records list these ongoing health conditions: {items}.",
        "no_conditions": "No ongoing health conditions are listed in your records.",
        "trend": "{name} has {moved} in each of your last {n} reports ({series} {unit}) and is now {where} the usual range ({range}).",
        "single": "Your latest {name} result was {value} {unit} ({date}), which is {where} the usual range ({range}).",
        "improving": " The latest change is towards the usual range.",
        "others": "Your other test results were within the usual range or are not usually a concern.",
        "moved": {"rising": "gone up", "falling": "gone down"},
        "where": {"high": "higher than", "low": "lower than", "near_high": "close to the top of",
                  "near_low": "close to the bottom of"},
        "bp_flag": "Your blood pressure was 140/90 or higher on two different days ({r1} on {d1} and {r2} on {d2}).",
        "bp_once": "Your latest blood pressure ({r1} on {d1}) was 140/90 or higher. It should be checked again on another day.",
        "bp_normal": "Your latest blood pressure ({r1} on {d1}) was below 140/90.",
        "hb_flag": "Your haemoglobin ({v} g/dL) is below the level used to check for anaemia ({c} g/dL).",
        "hb_normal": "Your haemoglobin ({v} g/dL) is not in the anaemia range.",
        "meds": "Medicines listed as ongoing: {items}.",
        "med_item": "{name} (since {year})",
        "no_meds": "No ongoing medicines are listed in your records.",
        "no_allergy": "No allergies are recorded. If you have any allergies, please tell your doctor.",
        "allergies": "Allergies recorded: {items}.",
        "ask_trend": "Why your {name} has been {moving}",
        "moving": {"rising": "going up", "falling": "going down"},
        "ask_value": "Your {name} result",
        "ask_bp": "Your blood pressure readings",
        "ask_hb": "Your haemoglobin level",
        "ask_allergy": "Any allergies you have, because none are recorded",
        "synthetic": "(This is a synthetic demonstration record, not a real patient.)",
        "disclaimer": "This summary only repeats what is in your records. It is not a diagnosis. "
                      "Please talk to your doctor before changing any medicine.",
    },
    "hi": {
        "titles": ["आपके रिकॉर्ड", "रिकॉर्ड में दर्ज बीमारियाँ", "ध्यान देने वाली जांचें",
                   "ब्लड प्रेशर और हीमोग्लोबिन", "आपकी दवाइयाँ और एलर्जी", "डॉक्टर से पूछने वाली बातें"],
        "overview": "आपके रिकॉर्ड में {first} से {last} तक की {n} रिपोर्ट हैं, जिनमें {k} लैब रिपोर्ट हैं।",
        "conditions": "आपके रिकॉर्ड में ये चल रही बीमारियाँ दर्ज हैं: {items}।",
        "no_conditions": "आपके रिकॉर्ड में कोई चल रही बीमारी दर्ज नहीं है।",
        "trend": "{name} आपकी पिछली {n} रिपोर्ट में लगातार {moved} है ({series} {unit}) और अब सामान्य सीमा ({range}) {where}।",
        "single": "{date} को आपकी {name} रिपोर्ट {value} {unit} थी, जो सामान्य सीमा ({range}) {where}।",
        "improving": " हाल का बदलाव सामान्य सीमा की ओर है।",
        "others": "आपकी बाकी जांचें सामान्य सीमा में थीं या आम तौर पर चिंता की बात नहीं हैं।",
        "moved": {"rising": "बढ़ा", "falling": "घटा"},
        "where": {"high": "से ज़्यादा है", "low": "से कम है", "near_high": "की ऊपरी हद के पास है",
                  "near_low": "की निचली हद के पास है"},
        "bp_flag": "आपका ब्लड प्रेशर दो अलग-अलग दिनों में 140/90 या उससे ज़्यादा था ({d1} को {r1} और {d2} को {r2})।",
        "bp_once": "आपका पिछला ब्लड प्रेशर ({d1} को {r1}) 140/90 या उससे ज़्यादा था। इसे किसी और दिन दोबारा जांचना चाहिए।",
        "bp_normal": "आपका पिछला ब्लड प्रेशर ({d1} को {r1}) 140/90 से कम था।",
        "hb_flag": "आपका हीमोग्लोबिन ({v} g/dL) एनीमिया की जांच वाली सीमा ({c} g/dL) से कम है।",
        "hb_normal": "आपका हीमोग्लोबिन ({v} g/dL) एनीमिया की सीमा में नहीं है।",
        "meds": "चल रही दवाइयाँ: {items}।",
        "med_item": "{name} ({year} से)",
        "no_meds": "आपके रिकॉर्ड में कोई चल रही दवा दर्ज नहीं है।",
        "no_allergy": "आपके रिकॉर्ड में कोई एलर्जी दर्ज नहीं है। अगर आपको कोई एलर्जी है, तो अपने डॉक्टर को ज़रूर बताएं।",
        "allergies": "दर्ज एलर्जी: {items}।",
        "ask_trend": "आपका {name} लगातार क्यों {moving} है",
        "moving": {"rising": "बढ़ रहा", "falling": "घट रहा"},
        "ask_value": "आपकी {name} की रिपोर्ट",
        "ask_bp": "आपके ब्लड प्रेशर की रीडिंग",
        "ask_hb": "आपका हीमोग्लोबिन",
        "ask_allergy": "आपको कोई एलर्जी है या नहीं, क्योंकि रिकॉर्ड में कोई एलर्जी दर्ज नहीं है",
        "synthetic": "(यह एक सिंथेटिक डेमो रिकॉर्ड है, किसी असली मरीज़ का नहीं।)",
        "disclaimer": "यह सारांश सिर्फ़ आपके रिकॉर्ड में लिखी बातों को दोहराता है। यह कोई निदान (डायग्नोसिस) नहीं है। "
                      "कोई भी दवा बदलने से पहले अपने डॉक्टर से बात करें।",
    },
}
SEVERITY_ORDER = {"high": 0, "moderate": 1, "low": 2, None: 3}


def cap(text):
    """Capital first letter, but leave names like 'eGFR' alone."""
    return text if len(text) > 1 and text[1].isupper() else text[0].upper() + text[1:]


def deviation(test):
    """How far the latest value is outside its range (0 when inside), used to rank alerts."""
    value, rng = test["latest"]["value"], test["reference_range"]
    if rng["high"] and value > rng["high"]:
        return (value - rng["high"]) / rng["high"]
    if rng["low"] and value < rng["low"]:
        return (rng["low"] - value) / rng["low"]
    return 0.0


def rank(tests):
    """Most serious first: by severity, then by distance outside the range."""
    return sorted(tests, key=lambda t: (SEVERITY_ORDER[t["severity"]], -deviation(t)))


def key_tests(analysis, limit=6):
    """The results a patient summary must explain: alerts (high, then moderate), most serious first.
    Used by the template AND required from Gemini, so code (not the AI) decides what is important."""
    return rank([t for t in analysis["trends"]["all_tests"] if t["severity"] in ("high", "moderate")])[:limit]


def nice_date(text, language="en"):
    """'2021-03-06' -> '06 Mar 2021' (English) or '06-03-2021' (Hindi)."""
    return date.fromisoformat(str(text)[:10]).strftime("%d-%m-%Y" if language == "hi" else "%d %b %Y")


def local_range(low, high, unit, language):
    """Reference range written in the summary's language."""
    if language != "hi":
        return range_text(low, high, unit)
    u = f" {unit}" if unit else ""
    if low is not None and high is not None:
        return f"{num(low)}-{num(high)}{u}"
    if high is not None:
        return f"{num(high)}{u} से कम"
    if low is not None:
        return f"{num(low)}{u} से ज़्यादा"
    return "कोई सीमा नहीं दी गई"


def s(text, fact_ids):
    return {"text": text, "fact_ids": [f for f in fact_ids if f]}


def _find(facts, section, key=None, value=None):
    return [f for f in facts if f.get("section") == section and (key is None or f.get(key) == value)]


def _ids(facts):
    return [f["id"] for f in facts]


# ----------------------------------------------------------------------------- patient summary
def patient_summary(analysis, facts, language="en"):
    T, plain = TEXT[language], PLAIN[language]
    sections = [{"title": t, "sentences": []} for t in T["titles"]]
    reports = analysis["reports"]
    labs = [r for r in reports if r["report_type"] == "lab report"]

    overview = _find(facts, "overview")
    sections[0]["sentences"].append(s(T["overview"].format(
        n=len(reports), k=len(labs), first=nice_date(reports[0]["report_date"], language),
        last=nice_date(reports[-1]["report_date"], language)), _ids(overview)))
    if analysis.get("synthetic"):
        sections[0]["sentences"].append(s(T["synthetic"], []))

    active = _find(facts, "conditions_active")
    if active:
        sections[1]["sentences"].append(s(T["conditions"].format(items=", ".join(f["name"] for f in active)), _ids(active)))
    else:
        sections[1]["sentences"].append(s(T["no_conditions"], []))

    # Test results that raised an alert (high and moderate), most serious first
    tests = analysis["trends"]["all_tests"]
    alerts = key_tests(analysis)
    asks = []
    for t in alerts:
        name = plain.get(t["test_name"], t["test_name"])
        rng = local_range(t["reference_range"]["low"], t["reference_range"]["high"], t["unit"], language)
        where = T["where"][t["range_status"]]
        points = t["previous"] + [t["latest"]]
        if t["direction"] in ("rising", "falling"):
            text = T["trend"].format(name=cap(name), moved=T["moved"][t["direction"]],
                                     n=len(points), series=" → ".join(num(p["value"]) for p in points),
                                     unit=t["unit"], where=where, range=rng)
            if t["trend_meaning"] == "worsening" and t["severity"] == "high":
                asks.append(s(T["ask_trend"].format(name=name, moving=T["moving"][t["direction"]]), t["_fact_ids"]))
        else:
            text = T["single"].format(name=name, value=num(t["latest"]["value"]), unit=t["unit"],
                                      date=nice_date(t["latest"]["test_date"], language), where=where, range=rng)
            if t["severity"] == "high":
                asks.append(s(T["ask_value"].format(name=name), t["_fact_ids"][:1]))
        if t["trend_meaning"] == "improving":
            text += T["improving"]
        # a sentence about a trend cites the trend fact; a sentence about one value cites only that value
        cited = t["_fact_ids"] if t["direction"] in ("rising", "falling") else t["_fact_ids"][:1]
        sections[2]["sentences"].append(s(text, cited))
    others = [t for t in tests if t not in alerts]
    if others:
        sections[2]["sentences"].append(s(T["others"], [t["_fact_ids"][0] for t in others]))

    # Screening
    bp = analysis["screening"]["high_blood_pressure"]
    bp_fact = _find(facts, "screening", "check", "bp")
    r = bp.get("readings_used", [])
    fmt = lambda i: {f"r{i + 1}": f"{r[i]['systolic']}/{r[i]['diastolic']}", f"d{i + 1}": nice_date(r[i]["date"], language)}
    if bp["status"] == "flag":
        sections[3]["sentences"].append(s(T["bp_flag"].format(**fmt(1), **fmt(0)), _ids(bp_fact)))
        asks.append(s(T["ask_bp"], _ids(bp_fact)))
    elif bp["status"] in ("raised_once", "normal") and r and str(r[0]["date"])[:4].isdigit():
        key = "bp_once" if bp["status"] == "raised_once" else "bp_normal"
        sections[3]["sentences"].append(s(T[key].format(**fmt(0)), _ids(bp_fact)))
        if bp["status"] == "raised_once":
            asks.append(s(T["ask_bp"], _ids(bp_fact)))
    an = analysis["screening"]["anaemia"]
    an_fact = _find(facts, "screening", "check", "anaemia")
    if an["status"] in ("flag", "normal"):
        key = "hb_flag" if an["status"] == "flag" else "hb_normal"
        sections[3]["sentences"].append(s(T[key].format(v=num(an["hemoglobin"]["value"]), c=num(an["cutoff"])),
                                          _ids(an_fact)))
        if an["status"] == "flag":
            asks.append(s(T["ask_hb"], _ids(an_fact)))

    # Medicines and allergies
    meds = analysis["medications"]["ongoing"]
    med_facts = _find(facts, "medications_ongoing")
    if meds:
        items = "; ".join(T["med_item"].format(name=m["name"], year=m["start_date"][:4]) for m in meds)
        sections[4]["sentences"].append(s(T["meds"].format(items=items), _ids(med_facts)))
    else:
        sections[4]["sentences"].append(s(T["no_meds"], []))
    if analysis["allergies"]:
        items = ", ".join(a["substance"] for a in analysis["allergies"])
        sections[4]["sentences"].append(s(T["allergies"].format(items=items), _ids(_find(facts, "allergies"))))
    else:
        sections[4]["sentences"].append(s(T["no_allergy"], _ids(_find(facts, "allergies"))))
        asks.append(s(T["ask_allergy"], _ids(_find(facts, "allergies"))))

    sections[5]["sentences"] = asks[:5]
    return {"language": language, "sections": [sec for sec in sections if sec["sentences"]],
            "disclaimer": T["disclaimer"]}


# ----------------------------------------------------------------------------- doctor summary
def doctor_summary(analysis, facts):
    p = analysis["profile"]
    sections = []

    def add(title, sentences):
        if sentences:
            sections.append({"title": title, "sentences": sentences})

    add("Patient and records", [s(f["text"], [f["id"]]) for f in _find(facts, "overview")])

    history = [s(f"Active: {f['name']} (since {f['text'].split('since ')[1].rstrip(').')}).", [f["id"]])
               for f in _find(facts, "conditions_active")]
    history += [s(f["text"].replace("Previous diagnosis: ", "Resolved: "), [f["id"]])
                for f in _find(facts, "conditions_resolved")]
    add("1. Documented history", history)

    tests = analysis["trends"]["all_tests"]
    alerts = rank([t for t in tests if t["alert"]])
    labs = []
    for t in alerts:
        rng = range_text(t["reference_range"]["low"], t["reference_range"]["high"], t["unit"])
        labs.append(s(f"{t['test_name']} {num(t['latest']['value'])} {t['unit']} ({t['latest']['test_date']}; "
                      f"ref {rng}): {t['range_status'].replace('_', ' ')} [{t['severity']} alert].", t["_fact_ids"][:1]))
    normal = [t for t in tests if not t["alert"]]
    if normal:
        labs.append(s("Within range or not of concern at the latest test: "
                      + ", ".join(f"{t['test_name']} {num(t['latest']['value'])} {t['unit']}" for t in normal) + ".",
                      [t["_fact_ids"][0] for t in normal]))
    add("2. Laboratory observations (latest values)", labs)

    changes = []
    for t in alerts:
        if t["direction"] in ("rising", "falling", "mixed") and len(t["_fact_ids"]) > 1:
            points = t["previous"] + [t["latest"]]
            changes.append(s(f"{t['test_name']}: {' -> '.join(num(x['value']) for x in points)} {t['unit']} "
                             f"({points[0]['test_date']} to {points[-1]['test_date']}); {t['direction']}"
                             f"{', ' + t['trend_meaning'] if t['trend_meaning'] in ('worsening', 'improving') else ''}.",
                             t["_fact_ids"][1:]))
    add("3. Changes over time", changes)

    meds = [s(f["text"].replace("Medication: ", ""), [f["id"]])
            for f in _find(facts, "medications_ongoing") + _find(facts, "medications_stopped")]
    meds += [s(f["text"], [f["id"]]) for f in _find(facts, "allergies")]
    add("4. Medications and allergies", meds)

    add("5. Rule-based screening (WHO cut-offs)", [s(f["text"], [f["id"]]) for f in _find(facts, "screening")])

    ml = []
    for f in _find(facts, "ml"):
        if f["kind"] == "ml_estimate":
            risk = analysis["risks"][f["disease"]]
            text = (f"{risk['label']}: {risk['risk_level']} (probability {num(risk['probability'])}, {risk['model']}). "
                    f"{risk['explanation'].get('summary', '')}")
            if f["already_documented"]:
                text += " Already a documented diagnosis: this estimate of undiagnosed risk is not meaningful here."
            ml.append(s(text, [f["id"]]))
        else:
            text = f["text"].replace("ML screening estimate for ", "")
            ml.append(s(text[0].upper() + text[1:], [f["id"]]))
    add("6. ML screening estimates (not diagnoses)", ml)

    missing = [s(f["text"], [f["id"]]) for f in facts if f["kind"] in ("missing", "data_note") and f["section"] != "ml"]
    not_calculated = [f for f in _find(facts, "ml") if f["kind"] == "missing"]
    if not_calculated:
        missing.append(s("Risk models not run because inputs are missing: "
                         + ", ".join(f["disease"] for f in not_calculated) + " (see the ML section).", _ids(not_calculated)))
    if not p.get("pregnant") and p.get("sex") == "female":
        missing.append(s("Pregnancy status is taken from the profile and was not verified in the reports.", _ids(_find(facts, "overview"))))
    add("7. Missing or unverified information", missing)

    add("Source reports", [s(f["text"], [f["id"]]) for f in _find(facts, "sources")])
    return {"sections": sections,
            "disclaimer": "Generated from the supplied records only. ML results are screening estimates from small "
                          "public datasets, not diagnoses. Clinical judgement is required."}


# ----------------------------------------------------------------------------- rendering
def render(summary, facts):
    """Plain text with the source report ids after every sentence."""
    report_ids = {f["id"]: f["report_ids"] for f in facts}
    lines = []
    for section in summary["sections"]:
        lines.append(section["title"].upper())
        for sentence in section["sentences"]:
            sources = sorted({r for fid in sentence["fact_ids"] for r in report_ids.get(fid, [])})
            lines.append("- " + sentence["text"] + (f" [{', '.join(sources)}]" if sources else ""))
        lines.append("")
    lines.append(summary["disclaimer"])
    return "\n".join(lines)


def template_summaries(analysis, facts=None, language="en"):
    facts = facts or build_facts(analysis)
    patient = patient_summary(analysis, facts, language)
    doctor = doctor_summary(analysis, facts)
    used = {fid for summ in (patient, doctor) for sec in summ["sections"] for x in sec["sentences"] for fid in x["fact_ids"]}
    sources = sorted({r for f in facts if f["id"] in used for r in f["report_ids"]})
    return {"method": "template", "language": language,
            "patient_summary": patient, "doctor_summary": doctor,
            "patient_summary_text": render(patient, facts), "doctor_summary_text": render(doctor, facts),
            "source_report_ids": sources}


if __name__ == "__main__":
    import json
    import sys

    from src.history_analysis.organise import analyse_history

    name = sys.argv[1] if len(sys.argv) > 1 else "patient_A_diabetes"
    language = sys.argv[2] if len(sys.argv) > 2 else "en"
    history = json.loads(open(f"datasets/synthetic_patients/{name}.json", encoding="utf-8").read())
    analysis = analyse_history(history)
    result = template_summaries(analysis, language=language)
    print("=" * 30, "PATIENT SUMMARY", "=" * 30)
    print(result["patient_summary_text"])
    print("\n" + "=" * 30, "DOCTOR SUMMARY", "=" * 30)
    print(result["doctor_summary_text"])
