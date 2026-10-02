import json


RESULTS_FILE = "results/master_research_results_FIXED.json"


def load_results(file_path=RESULTS_FILE):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def print_calibration_results(data):

    calibration = data["calibration"]

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)

    overall = calibration["overall"]

    print("\nOVERALL")
    print(f"Verifiable Samples : {overall['verifiable_samples']}")
    print(f"Coverage           : {overall['coverage']:.3f}")
    print(f"Brier Score        : {overall['brier_score']:.4f}")
    print(f"ECE                : {overall['ece']:.4f}")

    for setting in ["random", "popular", "adversarial"]:

        result = calibration[setting]

        print(f"\n{setting.upper()}")
        print(f"Verifiable Samples : {result['verifiable_samples']}")
        print(f"Coverage           : {result['coverage']:.3f}")
        print(f"Brier Score        : {result['brier_score']:.4f}")
        print(f"ECE                : {result['ece']:.4f}")


def print_dcs_results(data):

    dcs = data["dcs_counterfactual"]

    print("\n" + "=" * 60)
    print("DCS COUNTERFACTUAL RESULTS")
    print("=" * 60)

    print(f"Cases              : {dcs['cases']}")
    print(f"Answer Flips       : {dcs['answer_flips']}")
    print(f"Answer Flip Rate   : {dcs['answer_flip_rate']:.3f}")

    print("\nNote:")
    print(dcs["note"])


def print_ambiguity_results(data):

    ambiguity = data["ambiguity"]

    print("\n" + "=" * 60)
    print("AMBIGUITY / ABSTENTION RESULTS")
    print("=" * 60)

    print(
        f"Total Candidates       : "
        f"{ambiguity['total_candidates']}"
    )

    print(
        f"Positive Missed-Evidence: "
        f"{ambiguity['positive_missed_evidence']}"
    )

    print(
        f"Negative No-Evidence    : "
        f"{ambiguity['negative_no_evidence']}"
    )

    print(
        f"Abstention Rate         : "
        f"{ambiguity['abstention_rate']:.3f}"
    )

    print(
        f"False Contradictions    : "
        f"{ambiguity['false_contradictions']}"
    )


def print_gate_configuration(data):

    gate = data["frozen_gate"]

    print("\n" + "=" * 60)
    print("FROZEN VERIFIABILITY GATE")
    print("=" * 60)

    print(
        f"Minimum Confidence     : "
        f"{gate['min_confidence']}"
    )

    print(
        f"Maximum Area Ratio     : "
        f"{gate['max_area_ratio']}"
    )

    print(
        f"Minimum Cross-Source IoU: "
        f"{gate['min_cross_source_iou']}"
    )


def main():

    data = load_results()

    print("\nResearch results loaded successfully.")

    print_gate_configuration(data)

    print_calibration_results(data)

    print_dcs_results(data)

    print_ambiguity_results(data)


if __name__ == "__main__":
    main()