"""Automated tests for trend analysis. Run from the project folder: python -m pytest -v"""
from src.trends.trends import analyse_test, analyse_trends


def results(name, values, low=None, high=None, unit="mg/dL", units=None):
    dates = [f"2026-0{i + 1}-15" for i in range(len(values))]
    units = units or [unit] * len(values)
    return [{"report_id": f"R{i}", "test_name": name, "value": v, "unit": u,
             "ref_low": low, "ref_high": high, "test_date": d}
            for i, (v, u, d) in enumerate(zip(values, units, dates))]


def test_rising_hba1c_above_range_is_high_alert():
    a = analyse_test("HbA1c", results("HbA1c", [5.9, 6.4, 6.8], 4.0, 5.6, "%"))
    assert a["direction"] == "rising" and a["range_status"] == "high"
    assert a["trend_meaning"] == "worsening" and a["severity"] == "high"
    assert [p["value"] for p in a["previous"]] == [5.9, 6.4] and a["latest"]["value"] == 6.8


def test_rising_low_hemoglobin_is_improving_not_high():
    a = analyse_test("Hemoglobin", results("Hemoglobin", [10.0, 10.8, 11.5], 12.0, 15.5, "g/dL"))
    assert a["trend_meaning"] == "improving"
    assert a["severity"] == "moderate"          # still below range, but not worsening


def test_falling_low_hemoglobin_is_high_alert():
    a = analyse_test("Hemoglobin", results("Hemoglobin", [12.5, 11.8, 10.9], 12.0, 15.5, "g/dL"))
    assert a["direction"] == "falling" and a["severity"] == "high"


def test_stable_value_in_range_has_no_alert():
    a = analyse_test("Serum Creatinine", results("Serum Creatinine", [0.9, 0.91, 0.92], 0.6, 1.2))
    assert a["direction"] == "stable" and a["alert"] is False


def test_rising_towards_limit_is_moderate():
    a = analyse_test("SGPT", results("SGPT", [30, 36, 39], 7, 40, "U/L"))
    assert a["range_status"] == "near_high" and a["severity"] == "moderate"


def test_single_result_out_of_range_is_moderate():
    a = analyse_test("Glucose", results("Glucose", [130], 70, 100))
    assert a["direction"] == "not_enough_data" and a["severity"] == "moderate"


def test_far_outside_range_is_high_even_without_trend():
    a = analyse_test("Serum Creatinine", results("Serum Creatinine", [2.6], 0.6, 1.2))
    assert a["severity"] == "high"


def test_mixed_direction_out_of_range_is_moderate():
    a = analyse_test("HbA1c", results("HbA1c", [6.0, 6.8, 6.4], 4.0, 5.6, "%"))
    assert a["direction"] == "mixed" and a["severity"] == "moderate"


def test_only_latest_three_results_are_compared():
    a = analyse_test("HbA1c", results("HbA1c", [9.0, 8.0, 5.9, 6.4, 6.8], 4.0, 5.6, "%"))
    assert len(a["previous"]) == 2 and a["direction"] == "rising"


def test_no_reference_range_gives_no_alert():
    a = analyse_test("Vitamin X", results("Vitamin X", [10, 20, 30]))
    assert a["alert"] is False and a["trend_meaning"] == "unclear"


def test_different_units_are_not_mixed():
    a = analyse_test("Hemoglobin", results("Hemoglobin", [105, 11.0, 11.5], 12.0, 15.5,
                                           units=["g/L", "g/dL", "g/dL"]))
    assert len(a["previous"]) == 1 and "different unit" in a["note"]


def test_alerts_sorted_most_serious_first():
    data = results("SGPT", [37, 37.2, 37.4], 7, 40, "U/L") + results("HbA1c", [5.9, 6.4, 6.8], 4.0, 5.6, "%")
    out = analyse_trends(data)
    assert [a["severity"] for a in out["alerts"]] == ["high", "low"]
    
    
    

def test_low_bilirubin_is_not_an_alert():
    a = analyse_test("Total Bilirubin", results("Total Bilirubin", [0.4, 0.3, 0.1], 0.2, 1.2))
    assert a["range_status"] == "low" and a["alert"] is False and "Only high" in a["note"]


def test_falling_egfr_below_60_is_high_alert():
    a = analyse_test("eGFR", results("eGFR", [43.5, 34.5, 32.8], 60, None, "mL/min/1.73m2"))
    assert a["direction"] == "falling" and a["severity"] == "high"


def test_low_hemoglobin_still_alerts_because_both_sides_matter():
    a = analyse_test("Hemoglobin", results("Hemoglobin", [11.0], 12.0, 15.5, "g/dL"))
    assert a["alert"] is True


def test_big_swing_back_to_start_is_mixed_not_stable():
    a = analyse_test("Triglycerides", results("Triglycerides", [187.6, 131.4, 196.5], None, 150))
    assert a["direction"] == "mixed"