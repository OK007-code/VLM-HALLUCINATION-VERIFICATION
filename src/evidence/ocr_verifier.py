import easyocr


class OCRVerifier:
    def __init__(self):
        print("Loading OCR model...")
        self.reader = easyocr.Reader(["en"], gpu=False)
        print("OCR model loaded.")

    def extract_text(self, image_path):
        results = self.reader.readtext(image_path)

        detected_text = []

        for result in results:
            box, text, confidence = result

            detected_text.append({
                "text": text,
                "confidence": float(confidence),
                "box": box
            })

        return detected_text

    def verify(self, image_path, claimed_text):
        ocr_results = self.extract_text(image_path)

        claimed_text = claimed_text.lower().strip()

        best_match = None

        for result in ocr_results:
            detected_text = result["text"].lower().strip()

            if claimed_text in detected_text or detected_text in claimed_text:
                best_match = result
                break

        if best_match is None:
            return {
                "claimed_text": claimed_text,
                "detected_text": None,
                "confidence": 0.0,
                "status": "UNVERIFIED"
            }

        confidence = best_match["confidence"]

        if confidence >= 0.50:
            status = "SUPPORTED"
        else:
            status = "UNVERIFIED"

        return {
            "claimed_text": claimed_text,
            "detected_text": best_match["text"],
            "confidence": confidence,
            "status": status
        }


if __name__ == "__main__":

    verifier = OCRVerifier()

    image_path = "data/ocr_test.png"

    print("\nOCR TEST")
    print("=" * 60)

    results = verifier.extract_text(image_path)

    print("\nDetected Text:")

    for result in results:
        print(
            f"Text: {result['text']} | "
            f"Confidence: {result['confidence']:.4f}"
        )

    print("\nVerification Test:")

    test_result = verifier.verify(
        image_path=image_path,
        claimed_text="STOP"
    )

    print(test_result)