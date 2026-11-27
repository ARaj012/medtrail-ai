"""Tests for the grounding check and the Gemini pipeline, using SIMULATED Gemini answers
(no internet, no API key needed). Run from the project folder: python -m pytest -v"""
import copy
import json
from pathlib import Path

import pytest

from src.history_analysis.organise import analyse_history
from src.summaries.ai_summaries import (DOCTOR_LAYOUT, build_prompt, facts_for_ai, layout_for,
                                        privacy_problems, required_fact_ids, summaries)
from src.summaries.facts import build_facts
from src.summaries.gemini_client import GeminiBadAnswer, GeminiUnavailable, parse_json
from src.summaries.grounding import check_sentence, check_summary
from src.summaries.templates import key_tests, template_summaries

DATA = Path(__file__).resolve().parents[1] / "datasets" / "synthetic_patients"
PATIENTS = ["patient_A_diabetes", "patient_B_kidney", "patient_C_liver"]


def load(name):
    analysis = analyse_history(json.loads((DATA / f"{name}.json").read_text(encoding="utf-8")))
    return analysis, build_facts(analysis)


@pytest.fixture(scope="module")
def patient_a():
    analysis, facts = load("patient_A_diabetes")
    template = template_summaries(analysis, facts, "en")
    return analysis, facts, template


def fact(facts, kind, **match):
    return next(f for f in facts if f["kind"] == kind and all(f.get(k) == v for k, v in match.items()))


class FakeGemini:
    """Returns the prepared answers one after another and remembers the prompts it received."""
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt, system_instruction=None):
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return copy.deepcopy(answer)


def with_extra_sentence(summary, section_index, text, fact_ids):
    bad = copy.deepcopy(summary)
    bad["sections"][section_index]["sentences"].append({"text": text, "fact_ids": fact_ids})
    return bad


# ----------------------------------------------------------------------------- the check itself
@pytest.mark.parametrize("name", PATIENTS)
@pytest.mark.parametrize("language", ["en", "hi"])
def test_template_summaries_pass_every_check(name, language):
    analysis, facts = load(name)
    template = template_summaries(analysis, facts, language)
    for audience in ("patient", "doctor"):
        lang = language if audience == "patient" else "en"
        key_names = [t["test_name"] for t in key_tests(analysis)]
        check = check_summary(template[f"{audience}_summary"], facts, audience,
                              layout_for(audience, lang), required_fact_ids(facts, audience, key_names), key_names)
        assert check["passed"], check["problems"][:3]


def test_invented_number_is_caught(patient_a):
    _, facts, _ = patient_a
    hba1c = fact(facts, "lab_observation", test_name="HbA1c")
    by_id = {f["id"]: f for f in facts}
    assert check_sentence({"text": "Your latest HbA1c was 7 %.", "fact_ids": [hba1c["id"]]}, by_id, "patient") == []
    problems = check_sentence({"text": "Your latest HbA1c was 8.2 %.", "fact_ids": [hba1c["id"]]}, by_id, "patient")
    assert any("8.2" in p for p in problems)


def test_wrong_date_is_caught(patient_a):
    _, facts, _ = patient_a
    hba1c = fact(facts, "lab_observation", test_name="HbA1c")
    by_id = {f["id"]: f for f in facts}
    problems = check_sentence({"text": "HbA1c was 7 % on 2021-04-06.", "fact_ids": [hba1c["id"]]}, by_id, "doctor")
    assert any("date" in p for p in problems)
    ok = check_sentence({"text": "HbA1c was 7 % on 06 Mar 2021.", "fact_ids": [hba1c["id"]]}, by_id, "doctor")
    assert ok == []


def test_invented_condition_is_caught_in_english_and_hindi(patient_a):
    _, facts, _ = patient_a
    hba1c = fact(facts, "lab_observation", test_name="HbA1c")
    by_id = {f["id"]: f for f in facts}
    en = check_sentence({"text": "This suggests kidney disease.", "fact_ids": [hba1c["id"]]}, by_id, "patient")
    hi = check_sentence({"text": "यह किडनी की बीमारी दिखाता है।", "fact_ids": [hba1c["id"]]}, by_id, "patient")
    assert any("kidney disease" in p for p in en) and any("किडनी" in p for p in hi)


def test_documented_condition_is_allowed(patient_a):
    _, facts, _ = patient_a
    diabetes = next(f for f in facts if f["kind"] == "documented" and f["name"] == "Diabetes")
    by_id = {f["id"]: f for f in facts}
    assert check_sentence({"text": "Your records list diabetes since 2010.", "fact_ids": [diabetes["id"]]},
                          by_id, "patient") == []


def test_unknown_fact_id_and_uncited_claims_are_caught(patient_a):
    _, facts, _ = patient_a
    by_id = {f["id"]: f for f in facts}
    assert any("unknown" in p for p in check_sentence({"text": "Fine.", "fact_ids": ["F99"]}, by_id, "patient"))
    assert check_sentence({"text": "Your glucose was 101.6 mg/dL.", "fact_ids": []}, by_id, "patient")


def test_medicine_advice_is_caught(patient_a):
    _, facts, _ = patient_a
    metformin = next(f for f in facts if f["kind"] == "medication" and "Metformin" in f["text"])
    by_id = {f["id"]: f for f in facts}
    for text in ["You should stop taking metformin.", "Increase your metformin dose.", "Stop taking metformin."]:
        assert "gives medicine advice" in check_sentence({"text": text, "fact_ids": [metformin["id"]]}, by_id, "patient")
    ok = {"text": "Do not stop taking your medicines without talking to your doctor.", "fact_ids": []}
    assert check_sentence(ok, by_id, "patient") == []


def test_ml_estimates_kept_out_of_patient_summary_and_inside_doctor_ml_section(patient_a):
    _, facts, _ = patient_a
    ml = fact(facts, "ml_estimate", disease="diabetes")
    by_id = {f["id"]: f for f in facts}
    sentence = {"text": "The diabetes screening estimate is Low (probability 0.255).", "fact_ids": [ml["id"]]}
    assert check_sentence(sentence, by_id, "patient")
    assert check_sentence(sentence, by_id, "doctor", "6. ML screening estimates (not diagnoses)") == []
    assert any("outside the ML section" in p
               for p in check_sentence(sentence, by_id, "doctor", "2. Laboratory observations (latest values)"))
    unlabelled = {"text": "Diabetes risk is Low (0.255).", "fact_ids": [ml["id"]]}
    assert any("estimate" in p for p in check_sentence(unlabelled, by_id, "doctor", "6. ML screening estimates"))


def test_devanagari_digits_and_frightening_words_are_caught(patient_a):
    _, facts, _ = patient_a
    hba1c = fact(facts, "lab_observation", test_name="HbA1c")
    by_id = {f["id"]: f for f in facts}
    assert any("Devanagari" in p for p in check_sentence({"text": "HbA1c ७ % है।", "fact_ids": [hba1c["id"]]}, by_id, "patient"))
    assert any("frightening" in p for p in check_sentence({"text": "Your HbA1c of 7 % is dangerous.",
                                                           "fact_ids": [hba1c["id"]]}, by_id, "patient"))


def test_doctor_layout_and_coverage(patient_a):
    _, facts, template = patient_a
    doctor = template["doctor_summary"]
    required = required_fact_ids(facts, "doctor")
    # drop section 6 (ML): its ML facts are required, so coverage fails
    no_ml = copy.deepcopy(doctor)
    no_ml["sections"] = [s for s in no_ml["sections"] if not s["title"].startswith("6.")]
    check = check_summary(no_ml, facts, "doctor", DOCTOR_LAYOUT, required)
    assert not check["passed"] and any("leaves out" in p["problem"] for p in check["problems"])
    # a lab fact cited in the documented-history section
    lab = fact(facts, "lab_observation", test_name="Glucose")
    moved = with_extra_sentence(doctor, 1, lab["text"], [lab["id"]])
    assert any("belong in: 2. Laboratory" in p["problem"] for p in check_summary(moved, facts, "doctor", DOCTOR_LAYOUT)["problems"])
    renamed = copy.deepcopy(doctor)
    renamed["sections"][1]["title"] = "History"
    assert any("unexpected section" in p["problem"] for p in check_summary(renamed, facts, "doctor", DOCTOR_LAYOUT)["problems"])


def test_calling_out_of_range_results_within_range_is_caught(patient_a):
    # The sentence Gemini really wrote in the first run: Total Cholesterol, Triglycerides and
    # Serum Creatinine are ABOVE range, yet the sentence says results were within range.
    _, facts, _ = patient_a
    by_id = {f["id"]: f for f in facts}
    names = ["HDL Cholesterol", "Serum Creatinine", "Total Cholesterol", "Triglycerides"]
    trend_ids = [fact(facts, "trend", test_name=n)["id"] for n in names]
    text = ("Other test results such as HDL Cholesterol, Serum Creatinine, Total Cholesterol and Triglycerides "
            "were also checked, and many results were within their reference ranges.")
    problems = check_sentence({"text": text, "fact_ids": trend_ids}, by_id, "patient")
    assert any("within range" in p and "Total Cholesterol" in p for p in problems)
    # the honest version passes
    honest = "Your total cholesterol and triglycerides are higher than the usual range."
    ids = [fact(facts, "lab_observation", test_name=n)["id"] for n in ("Total Cholesterol", "Triglycerides")]
    assert check_sentence({"text": honest, "fact_ids": ids}, by_id, "patient") == []


def test_naming_a_test_without_citing_it_is_caught(patient_a):
    _, facts, _ = patient_a
    by_id = {f["id"]: f for f in facts}
    hba1c = fact(facts, "lab_observation", test_name="HbA1c")["id"]
    problems = check_sentence({"text": "Your HbA1c and Triglycerides were checked.", "fact_ids": [hba1c]}, by_id, "patient")
    assert any("Triglycerides" in p for p in problems)
    anaemia = fact(facts, "screening", check="anaemia")["id"]          # 'hemoglobin' vs 'Haemoglobin' in the fact
    assert check_sentence({"text": "Your hemoglobin is 15.6 g/dL.", "fact_ids": [anaemia]}, by_id, "patient") == []


def test_patient_summary_must_explain_the_results_chosen_by_code(patient_a):
    analysis, facts, template = patient_a
    key_names = [t["test_name"] for t in key_tests(analysis)]
    assert key_names[:2] == ["LDL Cholesterol", "HbA1c"] and len(key_names) == 6
    required = required_fact_ids(facts, "patient", key_names)
    missing_tg = copy.deepcopy(template["patient_summary"])
    tg = {fact(facts, kind, test_name="Triglycerides")["id"] for kind in ("lab_observation", "trend")}
    for section in missing_tg["sections"]:
        section["sentences"] = [x for x in section["sentences"] if not set(x["fact_ids"]) & tg]
    check = check_summary(missing_tg, facts, "patient", layout_for("patient", "en"), required)
    assert any("leaves out" in p["problem"] and "Triglycerides" in p["problem"] for p in check["problems"])


def test_layout_problem_says_where_the_fact_belongs(patient_a):
    _, facts, template = patient_a
    note = next(f for f in facts if f["kind"] == "data_note")
    doctor = template["doctor_summary"]
    in_labs = with_extra_sentence(doctor, 2, note["text"], [note["id"]])             # section 2: allowed
    assert check_summary(in_labs, facts, "doctor", DOCTOR_LAYOUT)["passed"]
    in_history = with_extra_sentence(doctor, 1, note["text"], [note["id"]])          # section 1: not allowed
    problems = [p["problem"] for p in check_summary(in_history, facts, "doctor", DOCTOR_LAYOUT)["problems"]]
    assert any("belong in: 2. Laboratory" in p and "7. Missing" in p for p in problems)


def test_prompt_lists_important_results_plain_names_and_required_facts(patient_a):
    analysis, facts, _ = patient_a
    key_names = [t["test_name"] for t in key_tests(analysis)]
    prompt = build_prompt(facts_for_ai(facts, analysis, "patient"), "patient", "en", key_names)
    hba1c = fact(facts, "lab_observation", test_name="HbA1c")["id"]
    assert "Important results" in prompt and f"- HbA1c: {hba1c}" in prompt
    assert "HbA1c: HbA1c (average blood sugar over the last few months)" in prompt
    assert f"{hba1c} [lab_observation, alert: high]" in prompt and "Facts you must cite:" in prompt
    doctor = build_prompt(facts_for_ai(facts, analysis, "doctor"), "doctor", "en", key_names)
    assert "6. ML screening estimates (not diagnoses)   (may cite: documented, missing, ml_estimate)" in doctor


# ----------------------------------------------------------------------------- mistakes from real Gemini runs
@pytest.fixture(scope="module")
def patient_b():
    analysis, facts = load("patient_B_kidney")
    return analysis, facts, template_summaries(analysis, facts, "hi")


def test_ml_section_may_explain_shap_without_repeating_estimate(patient_b):
    # Run 2, patient B: Gemini split the ML fact into several sentences; only the result needs "estimate"
    analysis, facts, template = patient_b
    ml = fact(facts, "ml_estimate", disease="diabetes")["id"]
    sentences = [
        "Type 2 diabetes screening estimate: Low risk (probability 0.26, Logistic Regression).",
        "SHAP analysis indicates Age (57) raised the risk most, while BMI (28.1) and Glucose (112.3) lowered the risk most.",
        "The model was trained on 768 women of Pima heritage aged 21+ and is less reliable for other groups.",
        "Diabetes is already documented, so this estimate is not meaningful for this patient.",
    ]
    doctor = copy.deepcopy(template["doctor_summary"])
    section = next(s for s in doctor["sections"] if s["title"].startswith("6."))
    section["sentences"] = [{"text": x, "fact_ids": [ml]} for x in sentences] + section["sentences"][1:]
    key_names = [t["test_name"] for t in key_tests(analysis)]
    check = check_summary(doctor, facts, "doctor", DOCTOR_LAYOUT, required_fact_ids(facts, "doctor", key_names))
    assert check["passed"], check["problems"]

    by_id = {f["id"]: f for f in facts}
    title = "6. ML screening estimates (not diagnoses)"
    no_label = {"text": "Type 2 diabetes: Low risk (probability 0.26).", "fact_ids": [ml]}
    assert any("without calling it a screening estimate" in p for p in check_sentence(no_label, by_id, "doctor", title))

    section["sentences"] = [{"text": x, "fact_ids": [ml]} for x in sentences[:3]] + section["sentences"][4:]
    problems = [p["problem"] for p in check_summary(doctor, facts, "doctor", DOCTOR_LAYOUT)["problems"]]
    assert any("not meaningful" in p for p in problems)          # the documented-diagnosis caveat is required


def test_time_span_not_in_facts_is_caught(patient_b):
    # Run 2, patient B (Hindi): "the last 3 reports" (Sep-Nov 2021) was turned into "the last 3 years"
    _, facts, _ = patient_b
    by_id = {f["id"]: f for f in facts}
    ids = [fact(facts, kind, test_name="Serum Creatinine")["id"] for kind in ("lab_observation", "trend")]
    wrong = ("क्रिएटिनिन (किडनी की जांच) का नवीनतम मान 2021-11-07 को 2.3 mg/dL था, और पिछले 3 वर्षों के दौरान "
             "यह 1.7 से 2.2 और फिर 2.3 mg/dL रहा है, जो बढ़ रहा है।")
    assert any("time span" in p for p in check_sentence({"text": wrong, "fact_ids": ids}, by_id, "patient"))
    english = "Your creatinine rose over the last 3 years from 1.7 to 2.3 mg/dL."
    assert any("time span" in p for p in check_sentence({"text": english, "fact_ids": ids}, by_id, "patient"))
    right = "पिछली 3 रिपोर्टों में क्रिएटिनिन 1.7 से 2.2 और फिर 2.3 mg/dL हुआ।"
    assert check_sentence({"text": right, "fact_ids": ids}, by_id, "patient") == []
    age = "यह सारांश 57 साल की महिला मरीज के रिकॉर्ड पर आधारित है।"            # age is a real time span in F01
    assert check_sentence({"text": age, "fact_ids": [fact(facts, "profile")["id"]]}, by_id, "patient") == []


def test_hindi_within_range_variants_are_caught(patient_b):
    _, facts, _ = patient_b
    by_id = {f["id"]: f for f in facts}
    creat = fact(facts, "lab_observation", test_name="Serum Creatinine")["id"]
    for text in ["अन्य सभी परिणाम सामान्य सीमा के भीतर रहे।", "ये परिणाम सामान्य रहे।"]:
        assert any("within range" in p for p in check_sentence({"text": text, "fact_ids": [creat]}, by_id, "patient"))


def test_patient_summary_must_say_whether_results_are_above_the_range():
    # Run 3, patient C (Hindi): LDL was given as "153 mg/dL ... रुझान मिला-जुला है" but never compared with its range
    analysis, facts = load("patient_C_liver")
    template = template_summaries(analysis, facts, "hi")
    key_names = [t["test_name"] for t in key_tests(analysis)]
    ldl = [fact(facts, kind, test_name="LDL Cholesterol")["id"] for kind in ("lab_observation", "trend")]
    real = ("LDL कोलेस्ट्रॉल ('खराब' कोलेस्ट्रॉल) 153 mg/dL है जो 2021-06-11 को लिया गया था, और पिछली 3 रिपोर्टों में "
            "यह 128.5 से 126.3 और 153 mg/dL (2020-06-05, 2021-04-15, 2021-06-11) रहा है तथा इसका रुझान मिला-जुला है।")
    fixed = "LDL कोलेस्ट्रॉल ('खराब' कोलेस्ट्रॉल) 153 mg/dL है, जो सामान्य सीमा (100 mg/dL से कम) से ज़्यादा है।"
    results = {}
    for label, text in (("real", real), ("fixed", fixed)):
        summary = copy.deepcopy(template["patient_summary"])
        section = summary["sections"][2]
        section["sentences"] = [x for x in section["sentences"] if not set(x["fact_ids"]) & set(ldl)]
        section["sentences"].insert(0, {"text": text, "fact_ids": ldl})
        summary["sections"][-1]["sentences"] = [x for x in summary["sections"][-1]["sentences"]
                                                if not set(x["fact_ids"]) & set(ldl)]
        results[label] = check_summary(summary, facts, "patient", layout_for("patient", "hi"),
                                       required_fact_ids(facts, "patient", key_names), key_names)
    assert any("LDL Cholesterol is above the usual range" in p["problem"] for p in results["real"]["problems"])
    assert results["fixed"]["passed"], results["fixed"]["problems"]


def test_patient_summary_must_list_every_condition_and_medicine():
    # Run 3, patient C (Hindi): 4 of 9 ongoing conditions and 2 of 3 ongoing medicines were mentioned
    analysis, facts = load("patient_C_liver")
    template = template_summaries(analysis, facts, "hi")
    key_names = [t["test_name"] for t in key_tests(analysis)]
    required = required_fact_ids(facts, "patient", key_names)
    conditions = [f["id"] for f in facts if f["section"] == "conditions_active"]
    medicines = [f["id"] for f in facts if f["section"] == "medications_ongoing"]
    assert set(conditions + medicines) <= set(required)
    partial = copy.deepcopy(template["patient_summary"])
    for section in partial["sections"]:
        for x in section["sentences"]:
            x["fact_ids"] = [f for f in x["fact_ids"] if f not in conditions[4:] + medicines[2:]]
    problems = [p["problem"] for p in check_summary(partial, facts, "patient", layout_for("patient", "hi"), required)["problems"]]
    assert sum("leaves out" in p for p in problems) == len(conditions[4:]) + len(medicines[2:])


def test_hindi_words_match_in_both_unicode_forms(patient_b):
    # Run 4, patient B (Hindi): "ज़्यादा" can be stored as ज + nukta (2 code points) or as one letter (U+095B)
    _, facts, _ = patient_b
    by_id = {f["id"]: f for f in facts}
    creat = fact(facts, "lab_observation", test_name="Serum Creatinine")["id"]
    for zya in ("\u091c\u093c", "\u095b"):
        text = f"क्रिएटिनिन 2.3 mg/dL है, जो सामान्य सीमा (0.6-1.2 mg/dL) से {zya}्यादा है।"
        summary = {"sections": [{"title": "x", "sentences": [{"text": text, "fact_ids": [creat]}]}]}
        assert check_summary(summary, facts, "patient", key_names=["Serum Creatinine"])["passed"]


def test_letters_from_another_script_are_caught(patient_b):
    # Run 4, patient B (Hindi): Gemini wrote the Urdu "می" instead of the Hindi "में"
    _, facts, _ = patient_b
    by_id = {f["id"]: f for f in facts}
    egfr = fact(facts, "lab_observation", test_name="eGFR")["id"]
    text = "आप अपने डॉक्टर से अपनी किडनी की जांच (eGFR) के परिणामों के बारे می बात कर सकते हैं।"
    assert any("ARABIC" in p for p in check_sentence({"text": text, "fact_ids": [egfr]}, by_id, "patient"))


def test_english_below_above_in_hindi_is_caught():
    # Run 4, patient C (Hindi): "(संदर्भ सीमा: below 100 mg/dL से कम)"
    analysis, facts = load("patient_C_liver")
    by_id = {f["id"]: f for f in facts}
    ldl = fact(facts, "lab_observation", test_name="LDL Cholesterol")["id"]
    bad = "LDL कोलेस्ट्रॉल 153 mg/dL (संदर्भ सीमा: below 100 mg/dL से कम) सामान्य सीमा से ज़्यादा है।"
    good = "LDL कोलेस्ट्रॉल 153 mg/dL है, जो सामान्य सीमा (100 mg/dL से कम) से ज़्यादा है।"
    assert any("English word" in p for p in check_sentence({"text": bad, "fact_ids": [ldl]}, by_id, "patient"))
    assert check_sentence({"text": good, "fact_ids": [ldl]}, by_id, "patient") == []


def test_rejected_gemini_answers_are_kept_for_evaluation(patient_a, tmp_path):
    analysis, facts, template = patient_a
    metformin = next(f for f in facts if f["kind"] == "medication" and "Metformin" in f["text"])
    bad = with_extra_sentence(template["patient_summary"], 4, "You should stop taking metformin.", [metformin["id"]])
    out = summaries(analysis, "en", generate=FakeGemini(bad, bad, template["doctor_summary"]), cache_dir=tmp_path)
    assert len(out["grounding_details"]["patient"]["rejected_ai_answers"]) == 2


def test_wrong_structure_fails():
    assert not check_summary({"summary": "text"}, [], "patient")["passed"]
    assert not check_summary("just text", [], "patient")["passed"]


# ----------------------------------------------------------------------------- privacy
@pytest.mark.parametrize("name", PATIENTS)
def test_gemini_never_sees_patient_id_facility_or_ml_facts(name):
    analysis, facts = load(name)
    facilities = {f["facility"] for f in facts if f.get("facility")}
    for audience in ("patient", "doctor"):
        ai_facts = facts_for_ai(facts, analysis, audience)
        prompt = build_prompt(ai_facts, audience, "en")
        assert analysis["patient_id"] not in prompt
        assert not any(fac in prompt for fac in facilities)
        assert privacy_problems(ai_facts) == []
        if audience == "patient":
            assert not any(f["kind"] == "ml_estimate" for f in ai_facts)


def test_privacy_filter_blocks_phone_email_aadhaar(patient_a, tmp_path):
    analysis, facts, _ = patient_a
    risky = copy.deepcopy(analysis)
    risky["data_notes"] = risky["data_notes"] + ["Contact 9876543210 or test@example.com, Aadhaar 1234 5678 9012."]
    fake = FakeGemini()
    out = summaries(risky, "en", generate=fake, cache_dir=tmp_path)
    assert fake.prompts == []                                  # Gemini was never called
    assert out["grounding_status"] == "fallback_template"
    assert "Privacy filter" in out["grounding_details"]["patient"]["reason"]


# ----------------------------------------------------------------------------- the pipeline
def test_good_answers_pass(patient_a, tmp_path):
    analysis, _, template = patient_a
    fake = FakeGemini(template["patient_summary"], template["doctor_summary"])
    out = summaries(analysis, "en", generate=fake, cache_dir=tmp_path)
    assert out["grounding_status"] == "passed"
    assert out["grounding_details"]["patient"]["method"] == "gemini"
    assert set(out) >= {"patient_summary", "doctor_summary", "source_report_ids", "grounding_status"}
    assert "A-R10" in out["source_report_ids"]
    assert "not a diagnosis" in out["patient_summary"]          # disclaimer added by code


def test_bad_answer_is_retried_with_the_problems(patient_a, tmp_path):
    analysis, facts, template = patient_a
    hba1c = fact(facts, "lab_observation", test_name="HbA1c")
    bad = with_extra_sentence(template["patient_summary"], 2, "Your HbA1c was 9.4 %.", [hba1c["id"]])
    fake = FakeGemini(bad, template["patient_summary"], template["doctor_summary"])
    out = summaries(analysis, "en", generate=fake, cache_dir=tmp_path)
    assert out["grounding_status"] == "passed_after_retry"
    assert "9.4" in fake.prompts[1] and "failed the automatic fact check" in fake.prompts[1]


def test_answer_failing_twice_falls_back_to_template(patient_a, tmp_path):
    analysis, facts, template = patient_a
    metformin = next(f for f in facts if f["kind"] == "medication" and "Metformin" in f["text"])
    bad = with_extra_sentence(template["patient_summary"], 4, "You should stop taking metformin.", [metformin["id"]])
    fake = FakeGemini(bad, bad, template["doctor_summary"])
    out = summaries(analysis, "en", generate=fake, cache_dir=tmp_path)
    assert out["grounding_status"] == "fallback_template"
    assert out["grounding_details"]["patient"]["method"] == "template"
    assert out["grounding_details"]["doctor"]["method"] == "gemini"
    assert "stop taking metformin" not in out["patient_summary"]


def test_gemini_unavailable_falls_back_to_template(patient_a, tmp_path):
    analysis, _, template = patient_a
    error = GeminiUnavailable("Gemini free-tier limit reached (error 429). Try again later.")
    out = summaries(analysis, "en", generate=FakeGemini(error, error), cache_dir=tmp_path)
    assert out["grounding_status"] == "fallback_template"
    assert "429" in out["grounding_details"]["doctor"]["reason"]
    assert out["patient_summary"] == template["patient_summary_text"]


def test_broken_json_gets_a_retry(patient_a, tmp_path):
    # Run 5, patient A: Gemini's answer was cut off; that is a failed attempt, not "Gemini unavailable"
    analysis, _, template = patient_a
    broken = GeminiBadAnswer('Gemini did not return valid JSON: {"sections": [{"title": "About your records"')
    fake = FakeGemini(broken, template["patient_summary"], template["doctor_summary"])
    out = summaries(analysis, "en", generate=fake, cache_dir=tmp_path)
    assert out["grounding_details"]["patient"]["status"] == "passed_after_retry"
    assert "not complete, valid JSON" in fake.prompts[1]
    out = summaries(analysis, "hi", generate=FakeGemini(broken, broken, template["doctor_summary"]), cache_dir=tmp_path)
    assert out["grounding_details"]["patient"]["status"] == "failed"          # then the template is used


def test_parse_json_accepts_fences_and_trailing_text():
    assert parse_json('```json\n{"sections": []}\n```') == {"sections": []}
    assert parse_json('{"sections": []}\nHope this helps!') == {"sections": []}
    with pytest.raises(GeminiBadAnswer):
        parse_json('{"sections": [{"title": "About your records"')


def test_template_only_mode(patient_a):
    analysis, _, _ = patient_a
    out = summaries(analysis, "hi", use_ai=False)
    assert out["grounding_status"] == "template_only"
    assert all(d["template_check_passed"] for d in out["grounding_details"].values())


def test_passed_answers_are_cached_and_rechecked(patient_a, tmp_path):
    analysis, _, template = patient_a
    summaries(analysis, "en", generate=FakeGemini(template["patient_summary"], template["doctor_summary"]),
              cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2
    fake = FakeGemini()                                        # would fail if called
    out = summaries(analysis, "en", generate=fake, cache_dir=tmp_path)
    assert fake.prompts == [] and out["grounding_status"] == "passed"
    assert out["grounding_details"]["patient"]["from_cache"] is True
