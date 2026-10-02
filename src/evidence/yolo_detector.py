from ultralytics import YOLO
from src.config import MIN_CONFIDENCE


class YOLODetector:

    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect(self, image_path):

        results = self.model(image_path, verbose=False)

        detections = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Apply frozen confidence threshold
                if confidence < MIN_CONFIDENCE:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = result.names[class_id]

                detections.append({
                    "object": class_name,
                    "confidence": confidence,
                    "box": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    ]
                })

        return detections


if __name__ == "__main__":

    detector = YOLODetector()

    image_path = "data/image.png"

    detections = detector.detect(image_path)

    print("\nYOLO DETECTIONS")
    print("=" * 60)

    for detection in detections:
        print(detection)