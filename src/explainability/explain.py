"""
SHAP explanations for the MedTrail disease models.

For one patient, SHAP splits the model's output into one contribution per input:
positive = pushed the risk up, negative = pushed it down. The contributions add up
exactly to (model output - baseline), where the baseline is the model's average output.

- Random Forest / XGBoost: shap.TreeExplainer (exact for tree models).
- Logistic Regression: shap.LinearExplainer on the scaled inputs.
If SHAP cannot explain a model, the result says so; no other method is labelled as SHAP.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression


def _split_pipeline(pipeline):
    """Separate the preparation steps (fill gaps, scale) from the final model."""
    return pipeline[:-1], pipeline[-1]


def _make_explainer(pipeline):
    prepare, model = _split_pipeline(pipeline)
    if isinstance(model, LogisticRegression):
        n_features = len(pipeline.feature_names_in_)
        # After StandardScaler the training data has mean 0 for every input,
        # so a single all-zero background row represents the average patient.
        background = shap.maskers.Independent(np.zeros((1, n_features)))
        return shap.LinearExplainer(model, background), "SHAP (LinearExplainer)", "log-odds"
    explainer = shap.TreeExplainer(model)
    unit = "log-odds" if type(model).__name__.startswith("XGB") else "probability"
    return explainer, "SHAP (TreeExplainer)", unit


def _positive_class(values, expected):
    """SHAP returns different shapes for different models; keep the 'has disease' class."""
    values = np.asarray(values)
    expected = np.atleast_1d(np.asarray(expected, dtype=float))
    if values.ndim == 3:            # (rows, inputs, classes)
        values = values[:, :, 1]
    if expected.size > 1:
        expected = expected[1:2]
    return values, float(expected[0])


def shap_values(pipeline, X):
    """SHAP contributions for every row of X (original, unscaled inputs)."""
    explainer, method, unit = _make_explainer(pipeline)
    prepare, _ = _split_pipeline(pipeline)
    X_prepared = prepare.transform(X[list(pipeline.feature_names_in_)])
    values, baseline = _positive_class(explainer.shap_values(X_prepared), explainer.expected_value)
    return values, baseline, method, unit


def explain_one(pipeline, X_row, labels=None, display=None, top=None):
    """Explanation for one patient, in a format the doctor's dashboard can display.

    labels:  readable input names, e.g. {"sc": "Serum Creatinine"}
    display: readable values where the raw number is unclear, e.g. {"sex": "female"}
    """
    labels, display = labels or {}, display or {}
    try:
        values, baseline, method, unit = shap_values(pipeline, X_row)
    except Exception as error:      # never break a prediction because an explanation failed
        return {"method": "unavailable", "message": f"SHAP explanation failed: {error}"}

    contributions = values[0]
    total = float(np.abs(contributions).sum()) or 1.0
    rows = []
    for column, contribution in zip(pipeline.feature_names_in_, contributions):
        value = float(X_row.iloc[0][column])
        rows.append({
            "input": column,
            "name": labels.get(column, column),
            "value": value,
            "display_value": display.get(column, f"{value:g}"),
            "contribution": round(float(contribution), 4),
            "direction": "raises risk" if contribution > 0 else "lowers risk" if contribution < 0 else "no effect",
            "share_percent": round(100 * abs(float(contribution)) / total, 1),
        })
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    if top:
        rows = rows[:top]

    raised = [f"{r['name']} ({r['display_value']})" for r in rows if r["contribution"] > 0][:2]
    lowered = [f"{r['name']} ({r['display_value']})" for r in rows if r["contribution"] < 0][:2]
    summary = []
    if raised:
        summary.append("Raised the risk most: " + ", ".join(raised) + ".")
    if lowered:
        summary.append("Lowered the risk most: " + ", ".join(lowered) + ".")

    return {
        "method": method,
        "unit": unit,
        "baseline": round(baseline, 4),
        "compared_with": "the average patient in the model's training data",
        "contributions": rows,
        "summary": " ".join(summary),
    }


def global_importance(pipeline, X):
    """Average size of each input's SHAP contribution across many patients (global importance)."""
    values, _, method, unit = shap_values(pipeline, X)
    importance = pd.Series(np.abs(values).mean(axis=0), index=pipeline.feature_names_in_)
    return importance.sort_values(ascending=False), method, unit