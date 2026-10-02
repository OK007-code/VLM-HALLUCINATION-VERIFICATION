import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)


class GroundingDINO:

    def __init__(
        self,
        model_name="IDEA-Research/grounding-dino-tiny"
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading Grounding DINO on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name
        ).to(self.device)

        print("Grounding DINO loaded.")


    def detect(self, image_path, object_name):

        image = Image.open(image_path).convert("RGB")

        # Important: period is intentional.
        text_prompt = object_name.strip().lower() + "."

        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.20,
            text_threshold=0.20,
            target_sizes=[image.size[::-1]]
        )[0]

        detections = []

        for score, label, box in zip(
            results["scores"],
            results["text_labels"],
            results["boxes"]
        ):

            detections.append({
                "object": label,
                "confidence": float(score),
                "box": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3])
                ]
            })

        return detections


if __name__ == "__main__":

    detector = GroundingDINO()

    image_path = "data/image.png"

    test_objects = [
        "cat",
        "bottle",
        "vase"
    ]

    for object_name in test_objects:

        print("\n" + "=" * 60)
        print("QUERY:", object_name)
        print("=" * 60)

        detections = detector.detect(
            image_path,
            object_name
        )

        for detection in detections:
            print(detection)