"""Automated tests for the screening rules. Run from the project folder: python -m pytest -v"""
from src.screening.screening import screen_anaemia, screen_blood_pressure


def hb(value, date="2026-10-01", unit="g/dL"):
    return {"report_id": "R1", "test_name": "Hemoglobin", "value": value, "unit": unit, "test_date": date}


ADULT_MAN = {"age": 40, "sex": "male"}
ADULT_WOMAN = {"age": 40, "sex": "female", "pregnant": False}


# ---------- anaemia ----------
def test_man_below_13_is_mild_anaemia():
    result = screen_anaemia(ADULT_MAN, [hb(12.5)])
    assert result["status"] == "flag" and result["severity"] == "mild"


def test_woman_12_5_is_normal_but_man_12_5_is_not():
    assert screen_anaemia(ADULT_WOMAN, [hb(12.5)])["status"] == "normal"
    assert screen_anaemia(ADULT_MAN, [hb(12.5)])["status"] == "flag"


def test_severity_boundaries():
    assert screen_anaemia(ADULT_WOMAN, [hb(11.0)])["severity"] == "mild"
    assert screen_anaemia(ADULT_WOMAN, [hb(10.9)])["severity"] == "moderate"
    assert screen_anaemia(ADULT_WOMAN, [hb(8.0)])["severity"] == "moderate"
    assert screen_anaemia(ADULT_WOMAN, [hb(7.9)])["severity"] == "severe"


def test_latest_result_is_used():
    result = screen_anaemia(ADULT_WOMAN, [hb(9.0, "2026-01-01"), hb(12.8, "2026-09-01")])
    assert result["status"] == "normal" and result["hemoglobin"]["value"] == 12.8


def test_g_per_litre_is_converted():
    assert screen_anaemia(ADULT_WOMAN, [hb(105, unit="g/L")])["hemoglobin"]["value"] == 10.5


def test_pregnancy_and_children_not_applicable():
    assert screen_anaemia({"age": 28, "sex": "female", "pregnant": True}, [hb(10)])["status"] == "not_applicable"
    assert screen_anaemia({"age": 10, "sex": "male"}, [hb(10)])["status"] == "not_applicable"


def test_missing_hemoglobin_or_sex():
    assert screen_anaemia(ADULT_WOMAN, [])["status"] == "needs_more_data"
    assert screen_anaemia({"age": 40}, [hb(10)])["status"] == "needs_more_data"


# ---------- blood pressure ----------
def bp(s, d, date):
    return {"systolic": s, "diastolic": d, "date": date, "source": "R1"}


def test_raised_on_two_days_is_flagged():
    result = screen_blood_pressure(ADULT_MAN, [bp(145, 85, "2026-10-01"), bp(130, 92, "2026-10-08")])
    assert result["status"] == "flag"


def test_raised_once_needs_repeat():
    assert screen_blood_pressure(ADULT_MAN, [bp(150, 95, "2026-10-01")])["status"] == "raised_once"


def test_two_readings_same_day_count_as_one_day():
    result = screen_blood_pressure(ADULT_MAN, [bp(150, 95, "2026-10-01"), bp(148, 94, "2026-10-01")])
    assert result["status"] == "raised_once"


def test_latest_normal_reading_is_normal():
    result = screen_blood_pressure(ADULT_MAN, [bp(150, 95, "2026-09-01"), bp(124, 80, "2026-10-01")])
    assert result["status"] == "normal"


def test_exactly_140_over_90_counts_as_raised():
    result = screen_blood_pressure(ADULT_MAN, [bp(140, 80, "2026-10-01"), bp(120, 90, "2026-10-05")])
    assert result["status"] == "flag"


def test_bp_not_applicable_for_pregnancy_and_children():
    assert screen_blood_pressure({"age": 30, "sex": "female", "pregnant": True}, [bp(150, 95, "2026-10-01")])["status"] == "not_applicable"
    assert screen_blood_pressure({"age": 16, "sex": "male"}, [bp(150, 95, "2026-10-01")])["status"] == "not_applicable"


def test_profile_bp_used_when_no_readings():
    result = screen_blood_pressure({"age": 50, "sex": "male", "systolic_bp": 152, "diastolic_bp": 96})
    assert result["status"] == "raised_once"


def test_no_bp_at_all():
    assert screen_blood_pressure(ADULT_MAN, [])["status"] == "needs_more_data"