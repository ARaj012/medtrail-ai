"""
Shared training steps for every MedTrail disease model.

The diabetes notebook shows each step written out in full. The kidney, liver and
heart notebooks call these functions instead, so every model is trained and
evaluated in exactly the same way.
"""
import json
import platform
from datetime import date

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
METRICS = ["roc_auc", "accuracy", "precision", "recall", "f1"]


def risk_level(probability):
    """Turn a probability into the risk level shown to doctors and patients."""
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Moderate"
    return "High"


def split_data(X, y):
    """80/20 split that keeps the same share of positive cases in both parts."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def make_models():
    """The three candidate pipelines. Filling gaps happens inside each pipeline (no leakage)."""
    return {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                             random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    eval_metric="logloss", random_state=RANDOM_STATE)),
        ]),
    }


def compare_models(models, X_train, y_train):
    """5-fold stratified cross-validation on the training data only. Best model first."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in models.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=METRICS)
        row = {"model": name}
        for metric in METRICS:
            row[metric] = scores[f"test_{metric}"].mean()
        row["roc_auc_spread"] = scores["test_roc_auc"].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False).round(3)


def test_best_model(cv_results, models, X_train, y_train, X_test, y_test, labels):
    """Fit the best cross-validated model and score it once on the test set."""
    best_name = cv_results.index[0]
    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    proba = best_model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
    print("Best model (highest cross-validated ROC-AUC):", best_name)
    print(f"Results on the {len(y_test)} test patients (used only once):")
    for name, value in test_metrics.items():
        print(f"  {name:<10}{value:.3f}")

    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=labels).plot(
        cmap="Blues", colorbar=False)
    plt.title(f"{best_name}: test set ({len(y_test)} patients)")
    plt.show()
    return best_name, best_model, proba, test_metrics


def risk_level_table(proba, y_test):
    """How many test patients fall in each risk level, and how many really had the disease."""
    check = pd.DataFrame({"risk_level": [risk_level(p) for p in proba], "had_disease": y_test.values})
    table = check.groupby("risk_level")["had_disease"].agg(patients="size", actually_had_disease="mean")
    table = table.reindex(["Low", "Moderate", "High"]).fillna(0)
    table["patients"] = table["patients"].astype(int)
    table["actually_had_disease"] = (table["actually_had_disease"] * 100).round().astype(int).astype(str) + "%"
    return table


def save_model(models_dir, disease, best_name, best_model, features, cv_results, test_metrics, notes):
    """Save the pipeline (.joblib) and a JSON record of how it was trained and how it scored."""
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / f"{disease}_pipeline.joblib")
    info = {
        "disease": disease,
        "model": best_name,
        "features": features,
        "risk_levels": {"Low": "below 0.30", "Moderate": "0.30 to below 0.60", "High": "0.60 and above"},
        "split": "80/20 stratified, random_state=42",
        "cross_validation": "5-fold stratified on the training set",
        "cv_results": cv_results.reset_index().to_dict(orient="records"),
        "test_metrics": {name: round(value, 3) for name, value in test_metrics.items()},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
        "trained_on": str(date.today()),
        **notes,
        "note": "Screening estimate from a small public dataset. Not a diagnosis.",
    }
    (models_dir / f"{disease}_metrics.json").write_text(json.dumps(info, indent=2))
    print("Saved:", f"{disease}_pipeline.joblib", "and", f"{disease}_metrics.json")