from src.evidence.detection_utils import calculate_iou


class CountVerifier:

    def _merge_detections(
        self,
        yolo_detections,
        dino_detections,
        iou_threshold=0.30
    ):
        """
        Merge YOLO and Grounding DINO detections.

        If YOLO and DINO boxes overlap sufficiently,
        they are treated as the same real-world object.
        """

        # Start with YOLO detections
        merged = []

        for detection in yolo_detections:
            merged.append({
                "object": detection["object"],
                "confidence": detection["confidence"],
                "box": detection["box"],
                "source": "YOLO"
            })

        # Add DINO detections only when they do not
        # correspond to an existing YOLO detection.
        for dino in dino_detections:

            matched = False

            for existing in merged:

                iou = calculate_iou(
                    dino["box"],
                    existing["box"]
                )

                if iou >= iou_threshold:
                    matched = True

                    # Keep the stronger detection
                    if dino["confidence"] > existing["confidence"]:
                        existing["confidence"] = dino["confidence"]
                        existing["box"] = dino["box"]

                    # Mark that both sources support this object
                    existing["source"] = "BOTH"

                    break

            # New DINO object
            if not matched:
                merged.append({
                    "object": dino["object"],
                    "confidence": dino["confidence"],
                    "box": dino["box"],
                    "source": "DINO"
                })

        return merged

    def verify(
        self,
        claim_count,
        yolo_detections,
        dino_detections
    ):

        # --------------------------------------------------
        # Merge both independent evidence sources
        # --------------------------------------------------

        merged_detections = self._merge_detections(
            yolo_detections,
            dino_detections,
            iou_threshold=0.30
        )

        detected_count = len(merged_detections)

        # --------------------------------------------------
        # Verdict
        # --------------------------------------------------

        if detected_count == claim_count:

            status = "SUPPORTED"

        elif detected_count == 0:

            status = "UNVERIFIED"

        else:

            status = "CONTRADICTED"

        return {
            "status": status,
            "claimed_count": claim_count,
            "detected_count": detected_count,

            # Useful for demo/debugging
            "yolo_count": len(yolo_detections),
            "dino_count": len(dino_detections),

            "merged_detections": merged_detections
        }