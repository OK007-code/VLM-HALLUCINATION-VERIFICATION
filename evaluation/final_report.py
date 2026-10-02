import json


RESULTS_FILE = "results/master_research_results_FIXED.json"


def load_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    data = load_results()

    print("\n" + "=" * 70)
    print("FINAL RESEARCH RESULTS REPORT")
    print("=" * 70)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = data["dataset"]

    print("\nDATASET")
    print("-" * 70)
    print(f"Dataset              : {dataset['name']}")
    print(f"Evaluation Samples   : {dataset['evaluation_samples']}")
    print(
        f"Settings             : "
        f"{', '.join(dataset['settings'])}"
    )

    # --------------------------------------------------
    # Frozen Gate
    # --------------------------------------------------

    gate = data["frozen_gate"]

    print("\nFROZEN VERIFIABILITY GATE")
    print("-" * 70)
    print(f"Minimum Confidence   : {gate['min_confidence']}")
    print(f"Maximum Area Ratio   : {gate['max_area_ratio']}")
    print(
        f"Minimum Cross-Source IoU : "
        f"{gate['min_cross_source_iou']}"
    )

    # --------------------------------------------------
    # Overall POPE
    # --------------------------------------------------

    overall = data["overall_results"]

    print("\nPOPE OVERALL")
    print("-" * 70)

    print(
        f"{'Metric':<15}"
        f"{'Value':>10}"
    )

    print("-" * 30)

    for metric in [
        "coverage",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr"
    ]:
        print(
            f"{metric:<15}"
            f"{overall[metric]:>10.4f}"
        )

    # --------------------------------------------------
    # Setting-wise
    # --------------------------------------------------

    print("\nPOPE SETTING-WISE")
    print("-" * 70)

    print(
        f"{'Setting':<15}"
        f"{'Coverage':>12}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>10}"
    )

    print("-" * 73)

    for setting, result in data["setting_wise_results"].items():

        print(
            f"{setting:<15}"
            f"{result['coverage']:>12.3f}"
            f"{result['accuracy']:>12.3f}"
            f"{result['precision']:>12.3f}"
            f"{result['recall']:>12.3f}"
            f"{result['f1']:>10.3f}"
        )

    # --------------------------------------------------
    # Ablation
    # --------------------------------------------------

    print("\nABLATION STUDY")
    print("-" * 70)

    print(
        f"{'Method':<25}"
        f"{'Coverage':>12}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>10}"
    )

    print("-" * 81)

    for method, result in data["ablation"].items():

        print(
            f"{method:<25}"
            f"{result['coverage']:>12.4f}"
            f"{result['accuracy']:>12.4f}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>10.4f}"
        )

    # --------------------------------------------------
    # Calibration
    # --------------------------------------------------

    calibration = data["calibration"]["overall"]

    print("\nCALIBRATION")
    print("-" * 70)

    print(
        f"Verifiable Samples : "
        f"{calibration['verifiable_samples']}"
    )

    print(
        f"Coverage           : "
        f"{calibration['coverage']:.4f}"
    )

    print(
        f"Brier Score        : "
        f"{calibration['brier_score']:.4f}"
    )

    print(
        f"ECE                : "
        f"{calibration['ece']:.4f}"
    )

    # --------------------------------------------------
    # DCS
    # --------------------------------------------------

    dcs = data["dcs_counterfactual"]

    print("\nDCS COUNTERFACTUAL EXPERIMENT")
    print("-" * 70)

    print(f"Cases              : {dcs['cases']}")
    print(f"Answer Flips       : {dcs['answer_flips']}")
    print(
        f"Answer Flip Rate   : "
        f"{dcs['answer_flip_rate']:.4f}"
    )

    # --------------------------------------------------
    # Ambiguity
    # --------------------------------------------------

    ambiguity = data["ambiguity"]

    print("\nAMBIGUITY / ABSTENTION")
    print("-" * 70)

    print(
        f"Total Candidates        : "
        f"{ambiguity['total_candidates']}"
    )

    print(
        f"Positive Missed Evidence: "
        f"{ambiguity['positive_missed_evidence']}"
    )

    print(
        f"Negative No Evidence    : "
        f"{ambiguity['negative_no_evidence']}"
    )

    print(
        f"Abstention Rate         : "
        f"{ambiguity['abstention_rate']:.4f}"
    )

    print(
        f"False Contradictions    : "
        f"{ambiguity['false_contradictions']}"
    )

    # --------------------------------------------------
    # External baseline
    # --------------------------------------------------

    baseline = data["external_baseline"]

    print("\nEXTERNAL BASELINE")
    print("-" * 70)

    print(f"Method       : {baseline['name']}")
    print(f"Source       : {baseline['source']}")
    print(f"Reproduced   : {baseline['reproduced']}")

    print("\n" + "=" * 70)
    print("END OF FINAL RESEARCH REPORT")
    print("=" * 70)


if __name__ == "__main__":
    main()