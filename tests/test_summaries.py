"""Tests for the fact sheet and template summaries, using the 3 synthetic demo patients.
Run from the project folder: python -m pytest -v"""
import json
import re
from pathlib import Path

import pytest

from src.history_analysis.organise import analyse_history
from src.summaries.facts import build_facts
from src.summaries.templates import template_summaries

PATIENTS = Path("datasets/synthetic_patients")


def load(name):
    return analyse_history(json.loads((PATIENTS / f"{name}.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module", params=["patient_A_diabetes", "patient_B_kidney", "patient_C_liver"])
def case(request):
    analysis = load(request.param)
    facts = build_facts(analysis)
    return analysis, facts, template_summaries(analysis, facts)


def sentences(summary):
    return [x for section in summary["sections"] for x in section["sentences"]]


def test_every_cited_fact_exists(case):
    _, facts, result = case
    ids = {f["id"] for f in facts}
    for summary in (result["patient_summary"], result["doctor_summary"]):
        for sentence in sentences(summary):
            assert set(sentence["fact_ids"]) <= ids


def test_every_source_report_exists(case):
    analysis, _, result = case
    known = {r["report_id"] for r in analysis["reports"]}
    assert result["source_report_ids"] and set(result["source_report_ids"]) <= known


def test_missing_allergies_are_stated_not_invented(case):
    _, _, result = case
    assert "No allergies are recorded" in result["patient_summary_text"]
    assert "No allergies are recorded" in result["doctor_summary_text"]


def test_doctor_summary_keeps_kinds_apart(case):
    _, _, result = case
    titles = [s["title"] for s in result["doctor_summary"]["sections"]]
    for expected in ["1. Documented history", "2. Laboratory observations (latest values)",
                     "6. ML screening estimates (not diagnoses)", "7. Missing or unverified information"]:
        assert expected in titles


def test_numbers_in_patient_alert_sentences_come_from_cited_facts(case):
    _, facts, result = case
    by_id = {f["id"]: f for f in facts}
    alerts = next(s for s in result["patient_summary"]["sections"] if s["title"] == "Test results to look at")
    for sentence in alerts["sentences"][:-1]:          # last line is the "other results" count
        text = re.sub(r"\(\d{2} \w{3} \d{4}\)", "", sentence["text"])          # remove dates
        text = re.sub(r"[A-Za-z]+/\S+", "", text)            # remove units such as mL/min/1.73m2
        allowed = {n for fid in sentence["fact_ids"] for n in by_id[fid]["numbers"]} | {1, 3}
        for token in re.findall(r"\d+(?:\.\d+)?", text):
            assert float(token) in allowed, f"{token} not in cited facts: {sentence['text']}"


def test_documented_diabetes_overrides_low_ml_estimate():
    result = template_summaries(load("patient_A_diabetes"))
    assert "Already a documented diagnosis" in result["doctor_summary_text"]


def test_kidney_patient_starts_with_creatinine():
    result = template_summaries(load("patient_B_kidney"))
    alerts = next(s for s in result["patient_summary"]["sections"] if s["title"] == "Test results to look at")
    assert alerts["sentences"][0]["text"].startswith("Creatinine")


def test_hindi_summary_is_in_hindi():
    result = template_summaries(load("patient_A_diabetes"), language="hi")
    text = result["patient_summary_text"]
    assert re.search(r"[ऀ-ॿ]", text) and "below" not in text
