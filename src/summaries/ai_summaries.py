"""
AI summaries with Gemini, checked by code, with the template summary as a safe fallback.

Flow for each summary (patient and doctor):
  1. Build the fact sheet (numbered facts F01, F02, ... from the analysed history).
  2. Remove anything Gemini does not need: patient id, facility names; patients never see ML facts.
     A privacy filter stops the call if an e-mail, phone or Aadhaar-like number is found.
  3. Gemini writes the summary as JSON; every sentence lists the fact ids it used.
  4. The grounding check tests every sentence (numbers, dates, report ids, conditions, advice,
     ML placement), the section layout, and that important facts are not left out.
  5. If it fails: one retry, with the list of problems. If it fails again, or Gemini is
     unavailable: the template summary is used. grounding_status says which happened.

Run from the project folder:
    python -m src.summaries.ai_summaries patient_A_diabetes en
    python -m src.summaries.ai_summaries patient_B_kidney hi --no-cache
    python -m src.summaries.ai_summaries patient_C_liver en --template-only
"""
import hashlib
import json
import re
from pathlib import Path

from src.summaries.facts import build_facts
from src.summaries.gemini_client import GeminiBadAnswer, GeminiUnavailable, generate_json, model_name
from src.summaries.grounding import check_summary
from src.summaries.templates import PLAIN, TEXT, key_tests, render, template_summaries

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "datasets" / "processed" / "summaries"
PROMPT_VERSION = "2026-11-27.5"         # change this when the prompts change, so old cached answers are not reused
MAX_ATTEMPTS = 2                        # first answer + one retry

PATIENT_KINDS = {"profile", "source_report", "documented", "medication", "allergy", "lab_observation",
                 "trend", "screening", "missing", "data_note"}           # no ml_estimate for patients

# Doctor summary: fixed sections, and which kinds of fact each section may cite.
DOCTOR_LAYOUT = [
    ("Patient and records", {"profile", "source_report"}),
    ("1. Documented history", {"documented"}),
    ("2. Laboratory observations (latest values)", {"lab_observation", "data_note"}),
    ("3. Changes over time", {"trend", "lab_observation", "data_note"}),
    ("4. Medications and allergies", {"medication", "allergy", "missing"}),
    ("5. Rule-based screening (WHO cut-offs)", {"screening"}),
    ("6. ML screening estimates (not diagnoses)", {"ml_estimate", "missing", "documented"}),
    ("7. Missing or unverified information", {"missing", "data_note", "profile"}),
    ("Source reports", {"source_report"}),
]

PII_PATTERNS = {
    "e-mail address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone number": re.compile(r"(?<![\d.])(?:\+91[ -]?)?[6-9]\d{9}(?![\d.])"),
    "Aadhaar-like number": re.compile(r"(?<![\d.])\d{4} ?\d{4} ?\d{4}(?![\d.])"),
}

SYSTEM_RULES = """You write summaries of medical records for MedTrail, a student project that uses synthetic data.
You receive numbered facts (F01, F02, ...). They are the ONLY information you may use.

Rules:
1. Every sentence must list, in "fact_ids", the ids of all facts it uses.
2. Copy numbers, units and dates exactly as they appear in the cited facts. Do not calculate anything new
   (no differences, percentages, averages, counts or unit conversions).
3. Do not add any diagnosis, condition, cause, test, medicine or number that is not in the cited facts.
   Do not explain test names with numbers (for example, do not write "over 3 months").
4. Never give treatment or medicine advice: never tell anyone to start, stop, take or change a medicine or dose.
5. Write all numbers with the digits 0-9, also in Hindi.
6. Mention a report id (like A-R10) only if it belongs to a fact the sentence cites.
7. Use the section titles exactly as given, in the given order.
8. Answer with JSON only, in this shape:
   {"sections": [{"title": "...", "sentences": [{"text": "...", "fact_ids": ["F01"]}]}]}"""

PATIENT_TASK = {
    "en": """Write a summary for the PATIENT in simple English that a person without medical training understands.
- Calm and kind; no frightening words (for example: dangerous, alarming, severe, critical).
- Do not diagnose. Describe conditions as written in the records ("your records list ...").
  Mention EVERY ongoing condition and EVERY ongoing medicine (you may group them in one or two sentences).
- Do not mention probabilities, risk models or AI predictions.
- In "Test results to look at", explain EACH result in the list "Important results" below, and no other single
  results. For each one, say clearly whether it is above or below the usual range (or close to a limit) and give
  the usual range. When you describe a change over time, cite both the latest-value fact and the trend fact.
  You may end with one sentence saying the other results did not raise an alert; do not name them or count them.
- Use the plain names given below for tests. Say "usual range" instead of "reference range".
- Write changes in words ("went up from 6.5 % to 6.8 % and then 7 %"), not with arrows.
- A trend fact covers the last few REPORTS, not years: never write a time span such as "over 3 years";
  use the dates in the fact instead ("between Sep 2021 and Nov 2021").
- In "Points to discuss with your doctor", suggest topics or questions to ask, never what to do about medicines.
- Write dates in the form 06 Mar 2021.
- At most 18 sentences.""",
    "hi": """Write a summary for the PATIENT in simple Hindi (Devanagari script) that a person without medical training understands.
- Calm and kind; no frightening words (for example: खतरनाक, गंभीर, जानलेवा).
- Do not diagnose. Describe conditions as written in the records ("आपके रिकॉर्ड में ... दर्ज है").
  Mention EVERY ongoing condition and EVERY ongoing medicine (you may group them in one or two sentences).
  Keep condition and medicine names as written, but write the rest of the sentence in Hindi: do not copy
  English phrases such as "(active, recorded since ...)"; write "(... से दर्ज)" instead.
- Do not mention probabilities, risk models or AI predictions.
- Keep test names such as HbA1c, eGFR, SGPT and units such as mg/dL in English letters.
- In the third section, explain EACH result in the list "Important results" below, and no other single results.
  For each one, say clearly whether it is above (सामान्य सीमा से ज़्यादा) or below (सामान्य सीमा से कम) the usual
  range, or close to a limit, and give the usual range. When you describe a change over time, cite both the
  latest-value fact and the trend fact.
  You may end with one sentence saying the other results did not raise an alert; do not name them or count them.
- Use the plain names given below for tests. Write changes in words, not with arrows.
- A trend fact covers the last few REPORTS (रिपोर्ट), not years: never write a time span such as "3 वर्षों" or
  "3 साल"; use the dates in the fact instead.
- In the last section, suggest topics or questions to ask the doctor, never what to do about medicines.
- Write numbers with 0-9 and dates as DD-MM-YYYY.
- Write usual ranges in Hindi: "below 100 mg/dL" -> "100 mg/dL से कम", "above 60" -> "60 से ज़्यादा".
  Do not use the English words "below" or "above" in Hindi sentences. Use only Devanagari and English letters
  (never Urdu/Arabic letters).
- At most 18 sentences.""",
}

DOCTOR_TASK = """Write a summary for a DOCTOR in concise clinical English.
- Keep these apart: documented history, laboratory observations, changes over time, medications and allergies,
  rule-based screening, ML screening estimates, and missing or unverified information.
- Section "1. Documented history" cites only documented facts; say "documented", not "diagnosed by MedTrail".
- Section "2. Laboratory observations (latest values)": latest values with date, reference range and status.
  You may group several tests in one sentence if you cite all their facts.
- ML results appear ONLY in section 6. The sentence that gives the risk level or probability must call it a
  "screening estimate" (not a diagnosis). You may add sentences for the SHAP explanation and the model note.
  If a fact says the condition is already documented, write that the estimate "is not meaningful" for this patient.
- Section 7 lists what is missing or unverified. Notes about the data (data_note facts) go in section 2, 3 or 7.
- Every fact listed under "Facts you must cite" must appear in the summary.
- Section "Source reports" lists the source reports briefly (you may group them).
- At most 40 sentences."""


# ----------------------------------------------------------------------------- facts for Gemini
def facts_for_ai(facts, analysis, audience):
    """Copies of the facts with only what Gemini needs: no patient id, no facility names,
    no ML facts for the patient summary."""
    out = []
    patient_id = analysis.get("patient_id") or ""
    for f in facts:
        if audience == "patient" and f["kind"] not in PATIENT_KINDS:
            continue
        text = f["text"]
        if patient_id:
            text = text.replace(patient_id, "The patient")
        if f.get("facility"):
            text = text.replace(f", {f['facility']})", ")")
        out.append({**{k: v for k, v in f.items() if k != "facility"}, "text": text})
    return out


def privacy_problems(facts):
    found = []
    for f in facts:
        for label, pattern in PII_PATTERNS.items():
            if pattern.search(f["text"]):
                found.append(f"{f['id']} contains a possible {label}")
    return found


def required_fact_ids(facts, audience, key_names=()):
    """Facts a summary must not leave out.
    Patient: the important results chosen by code (key_tests), every ongoing condition and medicine,
    and the allergy fact.
    Doctor: every result with an alert, the allergy fact, active conditions and ML estimates."""
    must = [f["id"] for f in facts if f["section"] == "allergies"]
    must += [f["id"] for f in facts if f["section"] in ("conditions_active", "medications_ongoing")]
    if audience == "patient":
        must += [f["id"] for f in facts if f["kind"] == "lab_observation" and f["test_name"] in key_names]
    else:
        must += [f["id"] for f in facts if f["kind"] == "lab_observation" and f.get("alert")]
        must += [f["id"] for f in facts if f["kind"] == "ml_estimate"]
    return must


def layout_for(audience, language):
    if audience == "doctor":
        return DOCTOR_LAYOUT
    return [(title, None) for title in TEXT[language]["titles"]]


def _tag(f):
    if f["kind"] in ("lab_observation", "trend"):
        return f"{f['kind']}, alert: {f['severity']}" if f.get("alert") else f"{f['kind']}, no alert"
    return f["kind"]


def build_prompt(facts, audience, language, key_names=()):
    task = DOCTOR_TASK if audience == "doctor" else PATIENT_TASK[language]
    parts = [task, "Section titles, in this order:"]
    for title, kinds in layout_for(audience, language):
        parts.append(f"- {title}" + (f"   (may cite: {', '.join(sorted(kinds))})" if kinds else ""))
    if audience == "patient":
        by_test = {}
        for f in facts:
            if f.get("test_name") in key_names:
                by_test.setdefault(f["test_name"], []).append(f["id"])
        parts.append("Important results (chosen by MedTrail's rules, most important first):")
        parts += [f"- {name}: {', '.join(by_test[name])}" for name in key_names if name in by_test]
        names = sorted({f["test_name"] for f in facts if f.get("test_name")})
        parts.append("Plain names for tests:")
        parts += [f"- {n}: {PLAIN[language][n]}" for n in names if n in PLAIN[language]]
    required = required_fact_ids(facts, audience, key_names)
    parts.append("Facts you must cite: " + ", ".join(required))
    parts.append("Facts:")
    parts += [f"{f['id']} [{_tag(f)}] {f['text']}" for f in facts]
    return "\n".join(parts)


def retry_prompt(prompt, problems):
    listed = "\n".join(f"- {p['problem']}" + (f" | sentence: {p['sentence']}" if p.get("sentence") else "")
                       for p in problems[:20])
    return (prompt + "\n\nYour previous answer failed the automatic fact check because of these problems:\n"
            + listed + "\nWrite the whole summary again and fix every problem. Follow all the rules.")


# ----------------------------------------------------------------------------- cache
def _cache_key(facts, audience, language):
    payload = json.dumps([facts, audience, language, model_name(), PROMPT_VERSION], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _cache_read(key, cache_dir):
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _cache_write(key, record, cache_dir):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / f"{key}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- one summary
def ai_summary(analysis, facts, audience, language="en", generate=None, use_cache=True, cache_dir=CACHE_DIR):
    """Ask Gemini for one summary and check it. Returns a dict with status
    'passed' | 'passed_after_retry' | 'failed' | 'unavailable' and the summary if it passed."""
    generate = generate or generate_json
    ai_facts = facts_for_ai(facts, analysis, audience)
    blocked = privacy_problems(ai_facts)
    if blocked:
        return {"status": "unavailable", "reason": "Privacy filter: " + "; ".join(blocked) + ". Gemini was not called.",
                "attempts": 0, "problems": []}

    key_names = [t["test_name"] for t in key_tests(analysis)]
    layout, required = layout_for(audience, language), required_fact_ids(ai_facts, audience, key_names)
    key = _cache_key(ai_facts, audience, language)
    if use_cache:
        cached = _cache_read(key, cache_dir)
        if cached:
            check = check_summary(cached["summary"], ai_facts, audience, layout, required, key_names)
            if check["passed"]:                       # re-checked, never trusted blindly
                return {**cached, "from_cache": True, "check": check}

    prompt = build_prompt(ai_facts, audience, language, key_names)
    problems, failed_answers = [], []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            answer = generate(prompt if attempt == 1 else retry_prompt(prompt, problems), SYSTEM_RULES)
        except GeminiBadAnswer as error:          # broken or cut-off JSON: counts as a failed attempt, so retry
            problems = [{"section": None, "sentence": None,
                         "problem": f"the answer was not complete, valid JSON ({str(error)[:160]})"}]
            failed_answers.append({"unreadable_answer": str(error)})
            continue
        except GeminiUnavailable as error:
            return {"status": "unavailable", "reason": str(error), "attempts": attempt - 1, "problems": problems}
        check = check_summary(answer, ai_facts, audience, layout, required, key_names)
        if check["passed"]:
            record = {"status": "passed" if attempt == 1 else "passed_after_retry", "attempts": attempt,
                      "model": model_name(), "prompt_version": PROMPT_VERSION, "summary": answer,
                      "problems_before_retry": problems}
            if use_cache:
                _cache_write(key, record, cache_dir)
            return {**record, "from_cache": False, "check": check}
        problems = check["problems"]
        failed_answers.append(answer)
    return {"status": "failed", "reason": f"Failed the fact check {MAX_ATTEMPTS} times.", "attempts": MAX_ATTEMPTS,
            "problems": problems, "failed_answers": failed_answers}


# ----------------------------------------------------------------------------- both summaries
def summaries(analysis, language="en", use_ai=True, generate=None, use_cache=True, cache_dir=CACHE_DIR):
    """The /ai/summary result: patient and doctor summaries, source report ids, grounding status."""
    facts = build_facts(analysis)
    template = template_summaries(analysis, facts, language)
    key_names = [t["test_name"] for t in key_tests(analysis)]
    result, details = {}, {}

    for audience in ("patient", "doctor"):
        lang = language if audience == "patient" else "en"
        fallback = template[f"{audience}_summary"]
        template_check = check_summary(fallback, facts, audience, layout_for(audience, lang),
                                       required_fact_ids(facts, audience, key_names), key_names)
        if use_ai:
            ai = ai_summary(analysis, facts, audience, lang, generate, use_cache, cache_dir)
        else:
            ai = {"status": "not_requested", "reason": "AI summaries were switched off.", "attempts": 0, "problems": []}

        if ai["status"] in ("passed", "passed_after_retry"):
            chosen = {**ai["summary"], "disclaimer": fallback["disclaimer"]}   # disclaimer always written by code
            details[audience] = {"method": "gemini", "status": ai["status"], "model": ai.get("model"),
                                 "attempts": ai["attempts"], "from_cache": ai.get("from_cache", False),
                                 "sentences": ai["check"]["sentences"], "supported": ai["check"]["supported"],
                                 "problems_before_retry": ai.get("problems_before_retry", [])}
        else:
            chosen = fallback
            details[audience] = {"method": "template", "status": ai["status"], "reason": ai.get("reason"),
                                 "attempts": ai["attempts"], "ai_problems": ai.get("problems", [])[:10],
                                 "rejected_ai_answers": ai.get("failed_answers", []),   # kept for the evaluation
                                 "template_check_passed": template_check["passed"],
                                 "sentences": template_check["sentences"], "supported": template_check["supported"]}
        result[audience] = chosen

    methods = {d["method"] for d in details.values()}
    if not use_ai:
        status = "template_only"
    elif methods == {"gemini"}:
        status = "passed_after_retry" if any(d["status"] == "passed_after_retry" for d in details.values()) else "passed"
    else:
        status = "fallback_template"

    used = {fid for s in result.values() for sec in s["sections"] for x in sec["sentences"] for fid in x["fact_ids"]}
    sources = sorted({r for f in facts if f["id"] in used for r in f["report_ids"]})
    return {
        "patient_id": analysis.get("patient_id"),
        "language": language,
        "patient_summary": render(result["patient"], facts),
        "doctor_summary": render(result["doctor"], facts),
        "patient_summary_sections": result["patient"],
        "doctor_summary_sections": result["doctor"],
        "source_report_ids": sources,
        "grounding_status": status,
        "grounding_details": details,
    }


PATIENT_FILES = ["patient_A_diabetes", "patient_B_kidney", "patient_C_liver"]
RESULTS_DIR = CACHE_DIR / "results"


def _print_details(out):
    print("grounding_status:", out["grounding_status"])
    for audience, d in out["grounding_details"].items():
        line = (f"  {audience}: method={d['method']}, status={d['status']}, attempts={d['attempts']}, "
                f"sentences supported={d['supported']}/{d['sentences']}")
        print(line + (", from cache" if d.get("from_cache") else ""))
        if d.get("reason"):
            print("    reason:", d["reason"])
        for label, items in (("fixed on retry", d.get("problems_before_retry", [])), ("problem", d.get("ai_problems", []))):
            for p in items:
                where = f"[{p['section']}] " if p.get("section") else ""
                print(f"    {label}: {where}{p['problem']} | {(p.get('sentence') or '')[:90]}")


if __name__ == "__main__":
    import sys
    import time

    from src.history_analysis.organise import analyse_history

    def load(name):
        path = ROOT / "datasets" / "synthetic_patients" / f"{name}.json"
        return analyse_history(json.loads(path.read_text(encoding="utf-8")))

    use_ai, use_cache = "--template-only" not in sys.argv, "--no-cache" not in sys.argv
    if "--all" in sys.argv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"{'patient':<20}{'lang':<6}{'grounding_status':<20}{'patient summary':<26}doctor summary")
        for name in PATIENT_FILES:
            for language in ("en", "hi"):
                out = summaries(load(name), language, use_ai=use_ai, use_cache=use_cache)
                (RESULTS_DIR / f"{name}_{language}.json").write_text(
                    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
                d = out["grounding_details"]
                cell = lambda x: f"{x['method']}/{x['status']}"
                print(f"{name:<20}{language:<6}{out['grounding_status']:<20}{cell(d['patient']):<26}{cell(d['doctor'])}")
                for audience, x in d.items():
                    if x["method"] == "template" and x["status"] != "not_requested":
                        print(f"    {audience} fell back: {x.get('reason')}")
                        for p in x.get("ai_problems", [])[:5]:
                            where = f"[{p['section']}] " if p.get("section") else ""
                            print(f"      - {where}{p['problem']} | {(p.get('sentence') or '')[:80]}")
                if use_ai and not all(x.get("from_cache") for x in d.values()):
                    time.sleep(5)               # be gentle with the free-tier rate limit
        print(f"\nFull results saved in {RESULTS_DIR.relative_to(ROOT)}")
        print("See one in detail with: python -m src.summaries.ai_summaries patient_B_kidney hi")
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        name = args[0] if args else "patient_A_diabetes"
        language = args[1] if len(args) > 1 else "en"
        out = summaries(load(name), language, use_ai=use_ai, use_cache=use_cache)
        _print_details(out)
        print("source_report_ids:", ", ".join(out["source_report_ids"]))
        print("\n" + "=" * 30, "PATIENT SUMMARY", "=" * 30)
        print(out["patient_summary"])
        print("\n" + "=" * 30, "DOCTOR SUMMARY", "=" * 30)
        print(out["doctor_summary"])
