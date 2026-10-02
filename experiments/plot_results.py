import json
from pathlib import Path

import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------
RESULTS_FILE = Path("results/master_research_results_FIXED.json")
FIGURE_DIR = Path("results/figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load results
# -----------------------------
with open(RESULTS_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)


# ============================================================
# 1. ABLATION STUDY
# ============================================================

ablation = data["ablation"]

methods = [
    "YOLO only",
    "DINO only",
    "YOLO + DINO"
]

coverage = [
    ablation["YOLO_only"]["coverage"] * 100,
    ablation["DINO_only"]["coverage"] * 100,
    ablation["YOLO_DINO_combined"]["coverage"] * 100
]

accuracy = [
    ablation["YOLO_only"]["accuracy"] * 100,
    ablation["DINO_only"]["accuracy"] * 100,
    ablation["YOLO_DINO_combined"]["accuracy"] * 100
]

f1 = [
    ablation["YOLO_only"]["f1"] * 100,
    ablation["DINO_only"]["f1"] * 100,
    ablation["YOLO_DINO_combined"]["f1"] * 100
]


x = range(len(methods))
width = 0.25

plt.figure(figsize=(9, 6))

plt.bar(
    [i - width for i in x],
    coverage,
    width,
    label="Coverage"
)

plt.bar(
    x,
    accuracy,
    width,
    label="Accuracy"
)

plt.bar(
    [i + width for i in x],
    f1,
    width,
    label="F1-score"
)

plt.xticks(list(x), methods)
plt.ylabel("Percentage (%)")
plt.title("Ablation Study: Evidence Source Comparison")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "ablation_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. POPE SETTING-WISE RESULTS
# ============================================================

settings = ["random", "popular", "adversarial"]

setting_results = data["setting_wise_results"]

coverage_setting = [
    setting_results[s]["coverage"] * 100
    for s in settings
]

accuracy_setting = [
    setting_results[s]["accuracy"] * 100
    for s in settings
]

f1_setting = [
    setting_results[s]["f1"] * 100
    for s in settings
]


x = range(len(settings))

plt.figure(figsize=(9, 6))

plt.plot(
    list(x),
    coverage_setting,
    marker="o",
    label="Coverage"
)

plt.plot(
    list(x),
    accuracy_setting,
    marker="o",
    label="Accuracy"
)

plt.plot(
    list(x),
    f1_setting,
    marker="o",
    label="F1-score"
)

plt.xticks(
    list(x),
    ["Random", "Popular", "Adversarial"]
)

plt.ylabel("Percentage (%)")
plt.title("POPE Setting-wise Performance")
plt.ylim(0, 100)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "pope_setting_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. CALIBRATION RESULTS
# ============================================================

calibration = data["calibration"]

brier = [
    calibration[s]["brier_score"]
    for s in settings
]

ece = [
    calibration[s]["ece"]
    for s in settings
]

x = range(len(settings))
width = 0.35

plt.figure(figsize=(8, 6))

plt.bar(
    [i - width / 2 for i in x],
    brier,
    width,
    label="Brier Score"
)

plt.bar(
    [i + width / 2 for i in x],
    ece,
    width,
    label="ECE"
)

plt.xticks(
    list(x),
    ["Random", "Popular", "Adversarial"]
)

plt.ylabel("Score")
plt.title("Calibration Results Across POPE Settings")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "calibration_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. OVERALL METRICS
# ============================================================

overall = data["overall_results"]

metric_names = [
    "Coverage",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "FPR"
]

metric_values = [
    overall["coverage"] * 100,
    overall["accuracy"] * 100,
    overall["precision"] * 100,
    overall["recall"] * 100,
    overall["f1"] * 100,
    overall["fpr"] * 100
]


plt.figure(figsize=(9, 6))

bars = plt.bar(
    metric_names,
    metric_values
)

plt.ylabel("Percentage (%)")
plt.title("Overall Performance on Held-out POPE Evaluation")
plt.ylim(0, 100)

for bar, value in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}",
        ha="center"
    )

plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "overall_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("All research figures generated successfully.")
print(f"Saved in: {FIGURE_DIR.resolve()}")

print("\nGenerated files:")
print("1. ablation_comparison.png")
print("2. pope_setting_performance.png")
print("3. calibration_results.png")
print("4. overall_performance.png")