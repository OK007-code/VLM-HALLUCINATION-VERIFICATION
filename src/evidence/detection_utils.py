def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width * intersection_height
    )

    area1 = (
        (box1[2] - box1[0]) *
        (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0]) *
        (box2[3] - box2[1])
    )

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def apply_nms(detections, iou_threshold=0.5):

    """
    Remove overlapping duplicate detections.

    Higher-confidence detections are kept first.
    """

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True
    )

    kept = []

    while detections:

        best = detections.pop(0)

        kept.append(best)

        remaining = []

        for detection in detections:

            iou = calculate_iou(
                best["box"],
                detection["box"]
            )

            # Keep only non-overlapping detections
            if iou < iou_threshold:
                remaining.append(detection)

        detections = remaining

    return kept


if __name__ == "__main__":

    detections = [
        {
            "object": "person",
            "confidence": 0.90,
            "box": [10, 10, 100, 100]
        },
        {
            "object": "person",
            "confidence": 0.80,
            "box": [15, 15, 95, 95]
        },
        {
            "object": "person",
            "confidence": 0.85,
            "box": [200, 200, 300, 300]
        }
    ]

    result = apply_nms(detections)

    print("\nNMS TEST")
    print("=" * 60)

    print("Original detections:", len(detections))
    print("After NMS:", len(result))

    for detection in result:
        print(detection)