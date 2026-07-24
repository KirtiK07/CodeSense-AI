import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from preprocess import get_train_test_split, load_and_clean

MODEL_PATH = "models/defect_model.pkl"

# Probability -> risk label thresholds.
# Both training (here) and inference (dashboard) must import these from the
# same place so they never drift out of sync.
RISK_THRESHOLDS = {
    "low_max": 0.33,     # predicted_prob < 0.33  -> Low
    "medium_max": 0.66,  # 0.33 <= predicted_prob < 0.66 -> Medium
                         # predicted_prob >= 0.66 -> High
}


def probability_to_risk(prob: float) -> str:
    if prob < RISK_THRESHOLDS["low_max"]:
        return "Low"
    if prob < RISK_THRESHOLDS["medium_max"]:
        return "Medium"
    return "High"


def train():
    X_train, X_test, y_train, y_test, feature_cols = get_train_test_split()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=None,   # SMOTE already balanced the training set
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Evaluation on held-out test set ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["Non-defective", "Defective"]))

    # Feature importance — useful for the dashboard's feature-importance chart.
    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("Top 5 most important features:")
    for name, score in importances[:5]:
        print(f"  {name}: {score:.3f}")

    # Medians from the ORIGINAL data (before SMOTE synthetic oversampling) — used as a
    # principled fallback in static_metrics.py for features that a given language's
    # static analyzer can't actually compute (e.g. Halstead metrics for non-Python
    # files, since Lizard doesn't expose operator/operand counts the way Radon does).
    X_full, _, _ = load_and_clean()
    feature_medians = X_full.median().to_dict()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": feature_cols,   # exact order the model expects at inference
        "feature_medians": feature_medians,
        "risk_thresholds": RISK_THRESHOLDS,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved trained model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    train()