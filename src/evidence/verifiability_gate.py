from src.config import (
    MIN_CONFIDENCE,
    MAX_AREA_RATIO,
    MIN_CROSS_SOURCE_IOU
)


class VerifiabilityGate:

    def __init__(
        self,
        min_confidence=MIN_CONFIDENCE,
        max_area_ratio=MAX_AREA_RATIO,
        min_cross_source_iou=MIN_CROSS_SOURCE_IOU
    ):
        self.min_confidence = min_confidence
        self.max_area_ratio = max_area_ratio
        self.min_cross_source_iou = min_cross_source_iou

    def check(self, evidence):

        yolo = evidence.get("yolo")
        dino = evidence.get("dino")
        iou = evidence.get("iou", 0.0)

        # No reliable evidence
        if yolo is None and dino is None:
            return {
                "gate": "AMBIGUOUS",
                "source": "NONE",
                "iou": iou
            }

        # Both sources available
        if yolo is not None and dino is not None:

            if iou >= self.min_cross_source_iou:
                return {
                    "gate": "VERIFIABLE",
                    "source": "BOTH",
                    "iou": iou
                }

            return {
                "gate": "VERIFIABLE_WITH_SINGLE_SOURCE",
                "source": "BOTH_LOW_IOU",
                "iou": iou
            }

        # Only YOLO
        if yolo is not None:
            return {
                "gate": "VERIFIABLE_WITH_SINGLE_SOURCE",
                "source": "YOLO",
                "iou": iou
            }

        # Only DINO
        if dino is not None:
            return {
                "gate": "VERIFIABLE_WITH_SINGLE_SOURCE",
                "source": "DINO",
                "iou": iou
            }

        return {
            "gate": "AMBIGUOUS",
            "source": "NONE",
            "iou": iou
        }


if __name__ == "__main__":

    gate = VerifiabilityGate()

    test_evidence = {
        "yolo": {
            "object": "cat",
            "confidence": 0.507,
            "area_ratio": 0.148
        },
        "dino": {
            "object": "cat",
            "confidence": 0.872,
            "area_ratio": 0.174
        },
        "iou": 0.8449
    }

    result = gate.check(test_evidence)

    print("\nVERIFIABILITY GATE")
    print("=" * 60)
    print("Gate:", result["gate"])
    print("Source:", result["source"])
    print("IoU:", result["iou"])