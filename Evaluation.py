"""
World Cup 2026 Match Predictor — Evaluation & Explainability
=============================================================
Input:  data/processed/match_features.csv
        models/xgboost.pkl
        models/random_forest.pkl
        models/logistic_regression.pkl
        models/scaler.pkl
        models/feature_columns.txt

Output: evaluation/confusion_matrix.png
        evaluation/calibration_curve.png
        evaluation/shap_summary.png
        evaluation/shap_bar.png
        evaluation/metrics_summary.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap

from sklearn.metrics import (
    accuracy_score, log_loss,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.calibration import calibration_curve

os.makedirs("evaluation", exist_ok=True)

LABELS      = ["Away win", "Draw", "Home win"]
CUTOFF      = pd.Timestamp("2022-01-01")
STYLE_COLOR = "#2d6a4f"   # consistent accent colour

# ── Load data & models ─────────────────────────────────────────────────────────

df = pd.read_csv("data/processed/match_features.csv", parse_dates=["date"])

with open("models/feature_columns.txt") as f:
    FEATURE_COLS = [line.strip() for line in f.readlines()]

test_df  = df[df["date"] >= CUTOFF].copy()
X_test   = test_df[FEATURE_COLS].fillna(0).values
y_test   = test_df["result"].values

with open("models/scaler.pkl",             "rb") as f: scaler = pickle.load(f)
with open("models/xgboost.pkl",            "rb") as f: xgb    = pickle.load(f)
with open("models/random_forest.pkl",      "rb") as f: rf     = pickle.load(f)
with open("models/logistic_regression.pkl","rb") as f: lr     = pickle.load(f)

X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": (lr,  X_test_scaled),
    "Random Forest":       (rf,  X_test),
    "XGBoost":             (xgb, X_test),
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. METRICS SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("METRICS SUMMARY")
print("=" * 60)

summary_rows = []
for name, (model, X) in models.items():
    y_pred      = model.predict(X)
    y_prob      = model.predict_proba(X)
    acc         = accuracy_score(y_test, y_pred)
    ll          = log_loss(y_test, y_prob)
    # Baseline: always predict most common class
    majority    = np.full_like(y_test, np.bincount(y_test).argmax())
    baseline    = accuracy_score(y_test, majority)
    summary_rows.append({"Model": name, "Accuracy": acc, "Log Loss": ll})
    print(f"\n{name}:")
    print(f"  Accuracy  : {acc:.4f}  (baseline: {baseline:.4f})")
    print(f"  Log loss  : {ll:.4f}")
    print(classification_report(y_test, y_pred, target_names=LABELS))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("evaluation/metrics_summary.csv", index=False)
print("\n✓ Saved evaluation/metrics_summary.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFUSION MATRICES  (one per model, side by side)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion matrices — hold-out test set (2022+)", fontsize=14, fontweight="bold")

for ax, (name, (model, X)) in zip(axes, models.items()):
    y_pred = model.predict(X)
    cm     = confusion_matrix(y_test, y_pred, normalize="true")
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(ax=ax, colorbar=False, cmap="Greens", values_format=".2f")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("evaluation/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved evaluation/confusion_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CALIBRATION CURVES  (are predicted probabilities trustworthy?)
# ══════════════════════════════════════════════════════════════════════════════
# A well-calibrated model: if it says "60% chance home wins", 60% of those
# matches should actually be home wins.

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Calibration curves — per outcome class", fontsize=14, fontweight="bold")

CLASS_COLORS = ["#e07b39", "#888", "#2d6a4f"]

for ax, (name, (model, X)) in zip(axes, models.items()):
    y_prob = model.predict_proba(X)
    for cls_idx, (cls_label, color) in enumerate(zip(LABELS, CLASS_COLORS)):
        y_bin          = (y_test == cls_idx).astype(int)
        prob_true, prob_pred = calibration_curve(y_bin, y_prob[:, cls_idx], n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", ms=4, label=cls_label, color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Mean predicted probability", fontsize=9)
    ax.set_ylabel("Fraction of positives", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("evaluation/calibration_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved evaluation/calibration_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SHAP FEATURE IMPORTANCE  (XGBoost only — most interpretable)
# ══════════════════════════════════════════════════════════════════════════════
# SHAP (SHapley Additive exPlanations) shows HOW MUCH each feature
# contributed to each individual prediction — not just global importance.

print("\nComputing SHAP values (XGBoost)...")

# Use a sample for speed — 500 rows is plenty for SHAP summary plots
sample_idx = np.random.choice(len(X_test), size=min(500, len(X_test)), replace=False)
X_sample   = X_test[sample_idx]

explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_sample)
# shap_values shape: (n_samples, n_features, n_classes)  for multi-class XGB

# ── 4a. SHAP summary plot (beeswarm) for "Home win" class ─────────────────────

shap.summary_plot(
    shap_values[:, :, 2],           # class 2 = home win
    X_sample,
    feature_names=FEATURE_COLS,
    plot_type="dot",
    show=False,
    plot_size=(10, 7),
)
plt.title("SHAP values — Home win probability\n(red = pushes prediction higher)", fontsize=12)
plt.tight_layout()
plt.savefig("evaluation/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved evaluation/shap_summary.png")

# ── 4b. Mean |SHAP| bar chart — global importance across all classes ──────────

mean_shap = np.mean(np.abs(shap_values).mean(axis=0), axis=1)  # avg over classes
shap_df   = pd.DataFrame({"feature": FEATURE_COLS, "importance": mean_shap})
shap_df   = shap_df.sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(shap_df["feature"], shap_df["importance"], color=STYLE_COLOR, height=0.6)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
ax.set_xlabel("Mean |SHAP value|  (average impact on model output)", fontsize=10)
ax.set_title("Feature importance — XGBoost (SHAP)", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("evaluation/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved evaluation/shap_bar.png")

# ── Print top 5 features ───────────────────────────────────────────────────────

print("\nTop 5 most influential features (SHAP):")
for _, row in shap_df.sort_values("importance", ascending=False).head(5).iterrows():
    print(f"  {row['feature']:<30} {row['importance']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. QUICK MANUAL PREDICTION CHECK
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SAMPLE PREDICTION CHECK (last 5 test matches)")
print("=" * 60)

for _, row in test_df.tail(5).iterrows():
    x      = np.array([[row[c] for c in FEATURE_COLS]])
    probs  = xgb.predict_proba(x)[0]
    actual = LABELS[int(row["result"])]
    pred   = LABELS[np.argmax(probs)]
    flag   = "✓" if actual == pred else "✗"
    print(f"\n  {row['home_team']} vs {row['away_team']}  ({row['date'].date()})")
    print(f"  Away win: {probs[0]:.2f}  Draw: {probs[1]:.2f}  Home win: {probs[2]:.2f}")
    print(f"  Predicted: {pred}  |  Actual: {actual}  {flag}")

print("\n✓ All evaluation outputs saved to evaluation/")
print("Next step: run app.py (Streamlit demo)")