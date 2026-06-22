"""
evaluate_crop_recommender.py
────────────────────────────
Evaluates the saved Crop Recommendation model.

Displays:
  1. Accuracy  (overall + per-class)
  2. Confusion Matrix  (ASCII grid)
  3. Feature Importances  (ranked bar chart)

The test set is re-created with the same random_state=42 used during
training, so results are reproducible and match the held-out test split.

Usage
  python backend:/scripts:/evaluate_crop_recommender.py
"""

import os
import json
import csv
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Path resolution (handles colon-suffixed workspace dirs) ───────────────────
def _find_workspace_root():
    candidate = os.path.abspath(os.path.dirname(__file__))
    for _ in range(10):
        for ds_name in ("datasets:", "datasets"):
            if os.path.isdir(os.path.join(candidate, ds_name)):
                return candidate, ds_name
        candidate = os.path.dirname(candidate)
    raise RuntimeError("Cannot locate workspace root.")

WORKSPACE, DS_DIR = _find_workspace_root()
MODELS_DIR_NAME   = "models:" if os.path.isdir(os.path.join(WORKSPACE, "models:")) else "models"

DATASET_PATH = os.path.join(WORKSPACE, DS_DIR,         "Crop_recommendation.csv")
MODEL_PATH   = os.path.join(WORKSPACE, MODELS_DIR_NAME, "crop_recommender.joblib")
CLASSES_PATH = os.path.join(WORKSPACE, MODELS_DIR_NAME, "crop_recommender_classes.json")

# ── Must match training config exactly ────────────────────────────────────────
FEATURES     = ["ph", "temperature", "humidity", "rainfall"]
TARGET       = "label"
RANDOM_STATE = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15


# ── Helpers ───────────────────────────────────────────────────────────────────

def div(title="", char="─", width=66):
    print(f"\n{char * width}")
    if title:
        print(f"  {title}")
        print(f"{char * width}")


def load_dataset(path):
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                X.append([float(row[f]) for f in FEATURES])
                y.append(row[TARGET].strip())
            except (ValueError, KeyError) as e:
                print(f"  [WARN] Skipping row: {e}")
    return X, y


def get_test_split(X, y_enc):
    """Reproduce the exact same held-out test split used in training."""
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - TRAIN_RATIO - VAL_RATIO,
        random_state=RANDOM_STATE,
    )
    _, test_idx = next(sss1.split(X, y_enc))
    X_test = [X[i] for i in test_idx]
    y_test = [y_enc[i] for i in test_idx]
    return X_test, y_test


# ── Section 1: Accuracy ───────────────────────────────────────────────────────

def show_accuracy(y_test, y_pred, classes):
    div("1. ACCURACY")
    overall = accuracy_score(y_test, y_pred)

    # Colour-code based on score
    def grade(score):
        if score >= 0.95: return "★★★  Excellent"
        if score >= 0.85: return "★★☆  Good"
        if score >= 0.70: return "★☆☆  Fair"
        return               "☆☆☆  Poor"

    print(f"  Overall Accuracy : {overall * 100:.2f}%   {grade(overall)}\n")

    # Per-class breakdown
    report = classification_report(
        y_test, y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    col_w = max(len(c) for c in classes) + 2
    header = f"  {'Crop':<{col_w}}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}  Grade"
    print(header)
    print(f"  {'─' * (col_w + 48)}")

    for cls in classes:
        m = report[cls]
        p, r, f1, sup = m["precision"], m["recall"], m["f1-score"], int(m["support"])
        g = grade(f1)
        flag = "  ◄ low" if f1 < 0.85 else ""
        print(f"  {cls:<{col_w}}  {p:>10.3f}  {r:>8.3f}  {f1:>8.3f}  {sup:>8}  {g}{flag}")

    print(f"\n  {'─' * (col_w + 48)}")
    ma = report["macro avg"]
    print(f"  {'Macro avg':<{col_w}}  {ma['precision']:>10.3f}  {ma['recall']:>8.3f}  {ma['f1-score']:>8.3f}")


# ── Section 2: Confusion Matrix ───────────────────────────────────────────────

def show_confusion_matrix(y_test, y_pred, classes):
    div("2. CONFUSION MATRIX")

    cm    = confusion_matrix(y_test, y_pred)
    n     = len(classes)
    col_w = max(len(c) for c in classes)
    cell  = 4   # width of each data cell

    # Short labels (first 4 chars) for column headers
    short = [c[:4] for c in classes]

    # Column header row
    pad = " " * (col_w + 3)
    print(pad + "  ".join(f"{s:>{cell}}" for s in short))
    print(pad + ("─" * cell + "  ") * n)

    total_correct = 0
    total_wrong   = 0

    for i, row_cls in enumerate(classes):
        row_vals = []
        for j, val in enumerate(cm[i]):
            if i == j:
                # Correct prediction — highlight
                cell_str = f"\033[32m{val:>{cell}}\033[0m"
                total_correct += val
            elif val > 0:
                # Wrong prediction — mark with colour
                cell_str = f"\033[31m{val:>{cell}}\033[0m"
                total_wrong += val
            else:
                cell_str = f"{val:>{cell}}"
            row_vals.append(cell_str)
        print(f"  {row_cls:<{col_w}} |" + "  ".join(row_vals))

    print(f"\n  ✅  Correct    : {total_correct}")
    print(f"  ❌  Misclassified : {total_wrong}")

    # Confusion pairs (off-diagonal non-zero)
    print(f"\n  Top confusion pairs (actual → predicted):")
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                pairs.append((cm[i][j], classes[i], classes[j]))
    pairs.sort(reverse=True)
    if pairs:
        for count, actual, predicted in pairs[:8]:
            print(f"    {actual:<18} → {predicted:<18}  ({count} sample{'s' if count > 1 else ''})")
    else:
        print("    None — perfect separation!")


# ── Section 3: Feature Importances ───────────────────────────────────────────

def show_feature_importances(pipeline, classes):
    div("3. FEATURE IMPORTANCES")

    rf         = pipeline.named_steps["clf"]
    importances = rf.feature_importances_
    n_trees    = rf.n_estimators

    # Sort descending
    ranked = sorted(zip(FEATURES, importances), key=lambda x: -x[1])

    print(f"  Model  : RandomForestClassifier  ({n_trees} trees)")
    print(f"  Metric : Mean Decrease in Impurity (Gini)\n")

    bar_max = 40
    print(f"  {'Feature':<15}  {'Importance':>10}  {'Rank':>5}  Bar")
    print(f"  {'─' * 15}  {'─' * 10}  {'─' * 5}  {'─' * bar_max}")

    medals = ["🥇", "🥈", "🥉"]
    for rank, (feat, imp) in enumerate(ranked, start=1):
        bar_len = int(imp * bar_max / max(importances))
        bar = "█" * bar_len
        medal = medals[rank - 1] if rank <= 3 else f"  #{rank}"
        print(f"  {feat:<15}  {imp:>10.4f}  {medal}   {bar}")

    # Tree-level variance insight
    print(f"\n  Per-tree importance variance (stability check):")
    tree_importances = [tree.feature_importances_ for tree in rf.estimators_]
    for i, feat in enumerate(FEATURES):
        vals = [t[i] for t in tree_importances]
        mean = sum(vals) / len(vals)
        std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        stability = "stable" if std / mean < 0.3 else "variable"
        print(f"    {feat:<15}  mean={mean:.4f}  std=±{std:.4f}  [{stability}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    div("CROP RECOMMENDATION — MODEL EVALUATION", char="═")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Dataset : {DATASET_PATH}")

    # Load model & metadata
    div("Loading Artefacts")
    pipeline = joblib.load(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        meta = json.load(f)
    classes = [meta["classes"][str(i)] for i in range(len(meta["classes"]))]
    print(f"  ✓ Pipeline loaded  ({type(pipeline.named_steps['clf']).__name__})")
    print(f"  ✓ {len(classes)} classes: {classes}")

    # Reconstruct test set
    div("Reconstructing Test Split")
    X, y = load_dataset(DATASET_PATH)
    le   = LabelEncoder()
    le.fit(y)
    y_enc   = list(le.transform(y))
    X_test, y_test = get_test_split(X, y_enc)
    print(f"  ✓ {len(X_test)} test samples  (stratified, random_state={RANDOM_STATE})")

    # Predict
    y_pred = pipeline.predict(X_test)

    # Show results
    show_accuracy(y_test, y_pred, classes)
    show_confusion_matrix(y_test, y_pred, classes)
    show_feature_importances(pipeline, classes)

    div("Evaluation Complete ✓", char="═")


if __name__ == "__main__":
    main()
