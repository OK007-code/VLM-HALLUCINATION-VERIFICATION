from PIL import Image

from src.claim_extractor import extract_and_normalize_claims

from src.evidence.yolo_detector import YOLODetector
from src.evidence.grounding_dino import GroundingDINO
from src.evidence.evidence_matcher import match_evidence
from src.evidence.verifiability_gate import VerifiabilityGate
from src.evidence.verdict import VerdictEngine
from src.evidence.count_verifier import CountVerifier
from src.evidence.spatial_verifier import SpatialVerifier
from src.evidence.color_verifier import ColorVerifier
from src.evidence.ocr_verifier import OCRVerifier


class VLMVerificationPipeline:

    def __init__(self):

        print("Loading verification pipeline...")

        self.yolo = YOLODetector()
        self.dino = GroundingDINO()

        self.gate = VerifiabilityGate()
        self.verdict_engine = VerdictEngine()

        self.count_verifier = CountVerifier()
        self.spatial_verifier = SpatialVerifier()
        self.color_verifier = ColorVerifier()
        self.ocr_verifier = OCRVerifier()

        print("Pipeline ready.")

    def verify(
        self,
        image_path,
        question,
        vlm_answer,
        extracted_claims=None
    ):

        # =====================================================
        # CLAIM EXTRACTION
        # =====================================================
        #
        # If claims are already extracted by demo.py,
        # use them directly.
        #
        # Otherwise extract them automatically from VLM answer.
        #

        if extracted_claims is not None:
            claims = extracted_claims
        else:
            claims = extract_and_normalize_claims(vlm_answer)

        # =====================================================
        # LOAD IMAGE
        # =====================================================

        image = Image.open(image_path).convert("RGB")

        image_width, image_height = image.size

        # =====================================================
        # RUN YOLO ONCE
        # =====================================================

        all_yolo_detections = self.yolo.detect(image_path)

        results = []

        # =====================================================
        # VERIFY EACH CLAIM
        # =====================================================

        for claim in claims:

            claim_type = claim["type"]
            data = claim["data"]

            # =================================================
            # OBJECT CLAIM
            # =================================================

            if claim_type == "object":

                object_name = data["object"]

                yolo_detections = [
                    detection
                    for detection in all_yolo_detections
                    if detection["object"] == object_name
                ]

                dino_detections = self.dino.detect(
                    image_path,
                    object_name
                )

                evidence = match_evidence(
                    yolo_detections,
                    dino_detections,
                    image_width,
                    image_height
                )

                gate_result = self.gate.check(evidence)

                verdict = self.verdict_engine.get_verdict(
                    claim,
                    gate_result
                )

                results.append({
                    "claim": claim,
                    "evidence": evidence,
                    "gate": gate_result,
                    "verdict": verdict
                })

            # =================================================
            # COUNT CLAIM
            # =================================================

            elif claim_type == "count":

                object_name = data["object"]
                claimed_count = data["count"]

                # YOLO detections
                yolo_detections = [
                    detection
                    for detection in all_yolo_detections
                    if detection["object"] == object_name
                ]

                # Grounding DINO detections
                dino_detections = self.dino.detect(
                    image_path,
                    object_name
                )

                # Combine YOLO + DINO
                count_result = self.count_verifier.verify(
                    claim_count=claimed_count,
                    yolo_detections=yolo_detections,
                    dino_detections=dino_detections
                )

                results.append({
                    "claim": claim,
                    "count_verification": count_result,
                    "verdict": count_result["status"]
                })

            # =================================================
            # SPATIAL CLAIM
            # =================================================

            elif claim_type == "spatial":

                object1 = data["object1"]
                relation = data["relation"]
                object2 = data["object2"]

                object1_detections = [
                    detection
                    for detection in all_yolo_detections
                    if detection["object"] == object1
                ]

                object2_detections = [
                    detection
                    for detection in all_yolo_detections
                    if detection["object"] == object2
                ]

                spatial_result = self.spatial_verifier.verify(
                    relation=relation,
                    object1_detections=object1_detections,
                    object2_detections=object2_detections
                )

                results.append({
                    "claim": claim,
                    "spatial_verification": spatial_result,
                    "verdict": spatial_result["status"]
                })

            # =================================================
            # COLOR CLAIM
            # =================================================

            elif claim_type == "color":

                object_name = data["object"]
                claimed_color = data["color"]

                yolo_detections = [
                    detection
                    for detection in all_yolo_detections
                    if detection["object"] == object_name
                ]

                # No detected object
                if not yolo_detections:

                    results.append({
                        "claim": claim,
                        "color_verification": {
                            "claimed_color": claimed_color,
                            "detected_color": None,
                            "confidence": 0.0,
                            "status": "UNVERIFIED"
                        },
                        "verdict": "UNVERIFIED"
                    })

                    continue

                # Use highest-confidence detection
                best_detection = max(
                    yolo_detections,
                    key=lambda x: x["confidence"]
                )

                color_result = self.color_verifier.verify(
                    image_path=image_path,
                    box=best_detection["box"],
                    claimed_color=claimed_color
                )

                results.append({
                    "claim": claim,
                    "color_verification": color_result,
                    "verdict": color_result["status"]
                })

            # =================================================
            # OCR / TEXT CLAIM
            # =================================================

            elif claim_type == "ocr":

                claimed_text = data["text"]

                ocr_result = self.ocr_verifier.verify(
                    image_path=image_path,
                    claimed_text=claimed_text
                )

                results.append({
                    "claim": claim,
                    "ocr_verification": ocr_result,
                    "verdict": ocr_result["status"]
                })

            # =================================================
            # UNKNOWN CLAIM TYPE
            # =================================================

            else:

                results.append({
                    "claim": claim,
                    "status": "NOT_IMPLEMENTED"
                })

        # =====================================================
        # FINAL OUTPUT
        # =====================================================

        return {
            "image": image_path,
            "question": question,
            "vlm_answer": vlm_answer,
            "claims": claims,
            "results": results
        }


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    pipeline = VLMVerificationPipeline()

    result = pipeline.verify(
        image_path="data/ocr_test.png",
        question="What does the sign say?",
        vlm_answer="There is a sign saying STOP."
    )

    print("\n")
    print("=" * 60)
    print("PIPELINE RESULT")
    print("=" * 60)

    print("\nVLM Answer:")
    print(result["vlm_answer"])

    print("\nClaims:")

    for claim in result["claims"]:
        print(claim)

    print("\nVerification Results:")

    for item in result["results"]:

        print("\nClaim:")
        print(item["claim"])

        # -------------------------------------------------
        # OBJECT
        # -------------------------------------------------

        if "gate" in item:

            print("Gate:")
            print(item["gate"])

            print("Evidence Source:")
            print(item["evidence"]["source"])

            print("IoU:")
            print(round(item["evidence"]["iou"], 4))

            print("Verdict:")
            print(item["verdict"])

        # -------------------------------------------------
        # COUNT
        # -------------------------------------------------

        elif "count_verification" in item:

            count_result = item["count_verification"]

            print("Claimed Count:")
            print(count_result["claimed_count"])

            print("YOLO Count:")
            print(count_result["yolo_count"])

            print("Grounding DINO Count:")
            print(count_result["dino_count"])

            print("Final Detected Count:")
            print(count_result["detected_count"])

            print("Verdict:")
            print(item["verdict"])

        # -------------------------------------------------
        # SPATIAL
        # -------------------------------------------------

        elif "spatial_verification" in item:

            spatial_result = item["spatial_verification"]

            print("Relation:")
            print(spatial_result.get("relation"))

            print("Verdict:")
            print(item["verdict"])

        # -------------------------------------------------
        # COLOR
        # -------------------------------------------------

        elif "color_verification" in item:

            color_result = item["color_verification"]

            print("Claimed Color:")
            print(color_result["claimed_color"])

            print("Detected Color:")
            print(color_result["detected_color"])

            print("Color Confidence:")
            print(round(color_result["confidence"], 4))

            print("Verdict:")
            print(item["verdict"])

        # -------------------------------------------------
        # OCR
        # -------------------------------------------------

        elif "ocr_verification" in item:

            ocr_result = item["ocr_verification"]

            print("Claimed Text:")
            print(ocr_result["claimed_text"])

            print("Detected Text:")
            print(ocr_result["detected_text"])

            print("OCR Confidence:")
            print(round(ocr_result["confidence"], 4))

            print("Verdict:")
            print(item["verdict"])

        # -------------------------------------------------
        # OTHER
        # -------------------------------------------------

        else:

            print("Status:")
            print(item["status"])