"""
train_crop_recommender.py
─────────────────────────
Trains a Random Forest classifier to recommend crops based on:
  Inputs  : ph, temperature, humidity, rainfall
  Output  : crop label (22 classes)

Pipeline
  1. RobustScaler  – outlier-resistant feature normalisation
  2. RandomForestClassifier (best params via GridSearchCV)

Outputs
  models:/crop_recommender.joblib        – serialised sklearn Pipeline
  models:/crop_recommender_classes.json  – label index → crop name mapping

Usage
  python backend:/scripts:/train_crop_recommender.py
"""

import os
import json
import csv
import joblib
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
# Walk upward from this file's directory until we find the workspace root
# (identified by the presence of a "datasets:" child directory).
def _find_workspace_root():
    candidate = os.path.abspath(os.path.dirname(__file__))
    for _ in range(10):
        # Check both with and without trailing colon (filesystem quirk)
        for ds_name in ("datasets:", "datasets"):
            if os.path.isdir(os.path.join(candidate, ds_name)):
                return candidate, ds_name
        candidate = os.path.dirname(candidate)
    raise RuntimeError("Cannot locate workspace root (no datasets/ dir found).")

WORKSPACE, DS_DIR_NAME = _find_workspace_root()

# Resolve models directory name the same way
MODELS_DIR_NAME = "models:" if os.path.isdir(os.path.join(WORKSPACE, "models:")) else "models"

DATASET_PATH = os.path.join(WORKSPACE, DS_DIR_NAME, "Crop_recommendation.csv")
MODELS_DIR   = os.path.join(WORKSPACE, MODELS_DIR_NAME)
MODEL_PATH   = os.path.join(MODELS_DIR, "crop_recommender.joblib")
CLASSES_PATH = os.path.join(MODELS_DIR, "crop_recommender_classes.json")

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES     = ["ph", "temperature", "humidity", "rainfall"]
TARGET       = "label"
RANDOM_STATE = 42

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO  = 0.15  (implicit)

PARAM_GRID = {
    "clf__n_estimators":      [100, 200, 300],
    "clf__max_depth":         [None, 10, 20],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf":  [1, 2],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset(path):
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                X.append([float(row[feat]) for feat in FEATURES])
                y.append(row[TARGET].strip())
            except (ValueError, KeyError) as e:
                print(f"  [WARN] Skipping row: {e}")
    return X, y


def stratified_split(X, y_enc, train_ratio, val_ratio, random_state):
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - train_ratio - val_ratio,
        random_state=random_state,
    )
    trainval_idx, test_idx = next(sss1.split(X, y_enc))

    X_tv = [X[i] for i in trainval_idx]
    y_tv = [y_enc[i] for i in trainval_idx]

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=random_state,
    )
    train_local, val_local = next(sss2.split(X_tv, y_tv))

    return (
        [trainval_idx[i] for i in train_local],
        [trainval_idx[i] for i in val_local],
        test_idx.tolist(),
    )


def subset(X, y, indices):
    return [X[i] for i in indices], [y[i] for i in indices]


def div(title=""):
    print(f"\n{'─' * 60}")
    if title:
        print(f"  {title}")
        print(f"{'─' * 60}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load ──────────────────────────────────────────────────────────────────
    div("1. Loading Dataset")
    print(f"  Path     : {DATASET_PATH}")
    X, y = load_dataset(DATASET_PATH)
    print(f"  Samples  : {len(X)}")
    print(f"  Features : {FEATURES}")
    counts = dict(sorted(Counter(y).items()))
    print(f"  Classes  : {len(counts)}")

    # 2. Encode labels ─────────────────────────────────────────────────────────
    div("2. Label Encoding")
    le      = LabelEncoder()
    y_enc   = list(le.fit_transform(y))
    classes = list(le.classes_)
    for i, cls in enumerate(classes):
        print(f"    {i:>2}  →  {cls}")

    # 3. Stratified split ──────────────────────────────────────────────────────
    div("3. Stratified Split  (70 / 15 / 15)")
    train_idx, val_idx, test_idx = stratified_split(
        X, y_enc, TRAIN_RATIO, VAL_RATIO, RANDOM_STATE
    )
    X_train, y_train = subset(X, y_enc, train_idx)
    X_val,   y_val   = subset(X, y_enc, val_idx)
    X_test,  y_test  = subset(X, y_enc, test_idx)
    print(f"  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)}   samples")
    print(f"  Test  : {len(X_test)}  samples")

    # 4. Build pipeline ────────────────────────────────────────────────────────
    div("4. Building Pipeline")
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("clf",    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    print("  RobustScaler  →  RandomForestClassifier")

    # 5. GridSearchCV on train+val ─────────────────────────────────────────────
    div("5. Hyperparameter Search  (GridSearchCV, 5-fold stratified CV)")
    X_tv = X_train + X_val
    y_tv = y_train + y_val

    grid = GridSearchCV(
        estimator  = pipeline,
        param_grid = PARAM_GRID,
        cv         = 5,
        scoring    = "accuracy",
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,
    )
    grid.fit(X_tv, y_tv)

    print(f"\n  Best CV Accuracy : {grid.best_score_ * 100:.2f}%")
    print("  Best Params:")
    for k, v in grid.best_params_.items():
        print(f"    {k:<35} = {v}")

    best = grid.best_estimator_

    # 6. Test set evaluation ───────────────────────────────────────────────────
    div("6. Test Set Evaluation")
    y_pred = best.predict(X_test)
    print(f"  Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=classes, digits=3))

    # 7. Feature importances ───────────────────────────────────────────────────
    div("7. Feature Importances")
    rf       = best.named_steps["clf"]
    feat_imp = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 50)
        print(f"  {feat:<15} {imp:.4f}  {bar}")

    # 8. Save artefacts ────────────────────────────────────────────────────────
    div("8. Saving Artefacts")
    joblib.dump(best, MODEL_PATH, compress=3)
    print(f"  Pipeline  →  {MODEL_PATH}")

    meta = {
        "features": FEATURES,
        "classes":  {str(i): cls for i, cls in enumerate(classes)},
    }
    with open(CLASSES_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata  →  {CLASSES_PATH}")

    # 9. Smoke test ────────────────────────────────────────────────────────────
    div("9. Smoke Test  (reload & predict)")
    loaded = joblib.load(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        m = json.load(f)

    # Representative sample: slightly acidic, warm, humid, moderate rain
    sample     = [[6.5, 28.0, 82.0, 120.0]]
    pred_idx   = loaded.predict(sample)[0]
    pred_proba = loaded.predict_proba(sample)[0]
    pred_label = m["classes"][str(pred_idx)]
    confidence = pred_proba[pred_idx] * 100

    print(f"  Input      :  ph=6.5  temp=28°C  humidity=82%  rainfall=120mm")
    print(f"  Prediction :  {pred_label}  ({confidence:.1f}% confidence)\n")

    top3 = sorted(enumerate(pred_proba), key=lambda x: -x[1])[:3]
    print("  Top-3 crops:")
    for idx, prob in top3:
        print(f"    {m['classes'][str(idx)]:<20}  {prob * 100:.1f}%")

    div("Done ✓")
    print(f"  Model ready at :  {MODEL_PATH}\n")


if __name__ == "__main__":
    main()
