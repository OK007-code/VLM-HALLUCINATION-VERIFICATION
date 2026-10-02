from src.config import MIN_CONFIDENCE, MAX_AREA_RATIO


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU)
    between two bounding boxes.

    Box format:
    [x1, y1, x2, y2]
    """

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


def calculate_area_ratio(box, image_width, image_height):
    """
    Calculate how much of the image
    is covered by the bounding box.
    """

    box_width = max(0, box[2] - box[0])
    box_height = max(0, box[3] - box[1])

    box_area = box_width * box_height
    image_area = image_width * image_height

    if image_area == 0:
        return 0.0

    return box_area / image_area


def get_best_evidence(
    detections,
    min_confidence=MIN_CONFIDENCE,
    max_area_ratio=MAX_AREA_RATIO,
    image_width=None,
    image_height=None
):
    """
    Select the strongest valid detection.

    For DINO, area ratio is also checked.
    """

    valid_detections = []

    for detection in detections:

        confidence = detection["confidence"]

        if confidence < min_confidence:
            continue

        if (
            image_width is not None
            and image_height is not None
        ):

            area_ratio = calculate_area_ratio(
                detection["box"],
                image_width,
                image_height
            )

            if area_ratio > max_area_ratio:
                continue

        else:
            area_ratio = None

        detection_copy = detection.copy()
        detection_copy["area_ratio"] = area_ratio

        valid_detections.append(detection_copy)

    if not valid_detections:
        return None

    return max(
        valid_detections,
        key=lambda x: x["confidence"]
    )


def match_evidence(
    yolo_detections,
    dino_detections,
    image_width,
    image_height
):
    """
    Combine YOLO and Grounding DINO evidence
    for the same object.
    """

    best_yolo = get_best_evidence(
        yolo_detections,
        image_width=image_width,
        image_height=image_height
    )

    best_dino = get_best_evidence(
        dino_detections,
        image_width=image_width,
        image_height=image_height
    )

    result = {
        "yolo": best_yolo,
        "dino": best_dino,
        "source": "NONE",
        "iou": 0.0
    }

    if best_yolo is not None and best_dino is not None:

        result["source"] = "BOTH"

        result["iou"] = calculate_iou(
            best_yolo["box"],
            best_dino["box"]
        )

    elif best_yolo is not None:

        result["source"] = "YOLO"

    elif best_dino is not None:

        result["source"] = "DINO"

    return result


if __name__ == "__main__":

    from PIL import Image
    from src.evidence.yolo_detector import YOLODetector
    from src.evidence.grounding_dino import GroundingDINO

    image_path = "data/image.png"

    # Load image dimensions
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    print("\nIMAGE")
    print("=" * 60)
    print("Path:", image_path)
    print("Size:", image_width, "x", image_height)

    # Load detectors
    yolo = YOLODetector()
    dino = GroundingDINO()

    # Object we want to verify
    object_name = "cat"

    print("\nQUERY:", object_name)

    # YOLO detection
    yolo_detections = [
        detection
        for detection in yolo.detect(image_path)
        if detection["object"] == object_name
    ]

    # Grounding DINO detection
    dino_detections = dino.detect(
        image_path,
        object_name
    )

    # Match both sources
    matched = match_evidence(
        yolo_detections,
        dino_detections,
        image_width,
        image_height
    )

    print("\nYOLO EVIDENCE")
    print("=" * 60)
    print(matched["yolo"])

    print("\nDINO EVIDENCE")
    print("=" * 60)
    print(matched["dino"])

    print("\nCOMBINED EVIDENCE")
    print("=" * 60)
    print("Source:", matched["source"])
    print("IoU:", round(matched["iou"], 4))