def classification_metrics(y_true, y_pred):
    tp = sum(
        1 for true, pred in zip(y_true, y_pred)
        if true == 1 and pred == 1
    )

    tn = sum(
        1 for true, pred in zip(y_true, y_pred)
        if true == 0 and pred == 0
    )

    fp = sum(
        1 for true, pred in zip(y_true, y_pred)
        if true == 0 and pred == 1
    )

    fn = sum(
        1 for true, pred in zip(y_true, y_pred)
        if true == 1 and pred == 0
    )

    total = len(y_true)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "samples": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr
    }


def coverage_metrics(total_samples, verifiable_samples):
    coverage = (
        verifiable_samples / total_samples
        if total_samples
        else 0.0
    )

    return {
        "total_samples": total_samples,
        "verifiable_samples": verifiable_samples,
        "coverage": coverage
    }


def brier_score(confidences, correct):
    if not confidences:
        return 0.0

    score = sum(
        (confidence - int(is_correct)) ** 2
        for confidence, is_correct
        in zip(confidences, correct)
    )

    return score / len(confidences)


def expected_calibration_error(
    confidences,
    correct,
    n_bins=5
):
    if not confidences:
        return 0.0

    ece = 0.0

    for bin_index in range(n_bins):

        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins

        if bin_index == n_bins - 1:
            indices = [
                i for i, confidence in enumerate(confidences)
                if lower <= confidence <= upper
            ]
        else:
            indices = [
                i for i, confidence in enumerate(confidences)
                if lower <= confidence < upper
            ]

        if not indices:
            continue

        bin_confidence = sum(
            confidences[i] for i in indices
        ) / len(indices)

        bin_accuracy = sum(
            int(correct[i]) for i in indices
        ) / len(indices)

        bin_weight = len(indices) / len(confidences)

        ece += bin_weight * abs(
            bin_accuracy - bin_confidence
        )

    return ece


if __name__ == "__main__":

    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 0, 1]

    print("\nMETRICS TEST")
    print("=" * 60)

    results = classification_metrics(
        y_true,
        y_pred
    )

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nCoverage:")

    coverage = coverage_metrics(
        total_samples=100,
        verifiable_samples=40
    )

    print(coverage)

    print("\nBrier Score:")

    confidences = [
        0.9,
        0.8,
        0.2,
        0.1,
        0.7
    ]

    correct = [
        True,
        True,
        False,
        True,
        True
    ]

    print(
        brier_score(
            confidences,
            correct
        )
    )

    print("\nECE:")

    print(
        expected_calibration_error(
            confidences,
            correct
        )
    )