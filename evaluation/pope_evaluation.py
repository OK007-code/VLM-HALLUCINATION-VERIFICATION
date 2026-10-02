import json


RESULTS_FILE = "results/master_research_results_FIXED.json"


def load_results(file_path=RESULTS_FILE):

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def print_overall_results(data):

    results = data["overall_results"]

    print("\n" + "=" * 60)
    print("POPE OVERALL RESULTS")
    print("=" * 60)

    print(f"Samples    : {results['samples']}")
    print(f"Coverage   : {results['coverage']:.3f}")
    print(f"Accuracy   : {results['accuracy']:.3f}")
    print(f"Precision  : {results['precision']:.3f}")
    print(f"Recall     : {results['recall']:.3f}")
    print(f"F1 Score   : {results['f1']:.3f}")
    print(f"FPR        : {results['fpr']:.3f}")


def print_setting_results(data):

    settings = data["setting_wise_results"]

    print("\n" + "=" * 60)
    print("SETTING-WISE RESULTS")
    print("=" * 60)

    for setting, results in settings.items():

        print(f"\n{setting.upper()}")

        print(f"Samples    : {results['samples']}")
        print(f"Coverage   : {results['coverage']:.3f}")
        print(f"Accuracy   : {results['accuracy']:.3f}")
        print(f"Precision  : {results['precision']:.3f}")
        print(f"Recall     : {results['recall']:.3f}")
        print(f"F1 Score   : {results['f1']:.3f}")
        print(f"FPR        : {results['fpr']:.3f}")


def print_ablation_results(data):

    ablation = data["ablation"]

    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)

    for method, results in ablation.items():

        print(f"\n{method}")

        print(f"Coverage   : {results['coverage']:.4f}")
        print(f"Accuracy   : {results['accuracy']:.4f}")
        print(f"Precision  : {results['precision']:.4f}")
        print(f"Recall     : {results['recall']:.4f}")
        print(f"F1 Score   : {results['f1']:.4f}")


def main():

    data = load_results()

    print("\nMaster research results loaded successfully.")

    print_overall_results(data)

    print_setting_results(data)

    print_ablation_results(data)


if __name__ == "__main__":
    main()