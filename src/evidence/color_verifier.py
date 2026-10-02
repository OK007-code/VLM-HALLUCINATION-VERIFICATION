import cv2
import numpy as np


class ColorVerifier:

    def estimate_color(self, image_path, box):
        image = cv2.imread(image_path)

        if image is None:
            return {
                "color": None,
                "confidence": 0.0
            }

        x1, y1, x2, y2 = map(int, box)

        height, width = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return {
                "color": None,
                "confidence": 0.0
            }

        # Use central 60% of the bounding box
        box_width = x2 - x1
        box_height = y2 - y1

        margin_x = int(box_width * 0.20)
        margin_y = int(box_height * 0.20)

        crop = image[
            y1 + margin_y:y2 - margin_y,
            x1 + margin_x:x2 - margin_x
        ]

        if crop.size == 0:
            return {
                "color": None,
                "confidence": 0.0
            }

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Ignore dark and low-saturation pixels
        valid_pixels = (saturation > 40) & (value > 50)

        if np.sum(valid_pixels) == 0:
            return {
                "color": "unknown",
                "confidence": 0.0
            }

        hue = hsv[:, :, 0][valid_pixels]

        color_counts = {
            "red": 0,
            "orange": 0,
            "yellow": 0,
            "green": 0,
            "blue": 0,
            "purple": 0,
            "pink": 0
        }

        for h in hue:

            if h <= 10 or h >= 170:
                color_counts["red"] += 1

            elif h <= 20:
                color_counts["orange"] += 1

            elif h <= 35:
                color_counts["yellow"] += 1

            elif h <= 85:
                color_counts["green"] += 1

            elif h <= 130:
                color_counts["blue"] += 1

            elif h <= 160:
                color_counts["purple"] += 1

            else:
                color_counts["pink"] += 1

        total = sum(color_counts.values())

        if total == 0:
            return {
                "color": "unknown",
                "confidence": 0.0
            }

        detected_color = max(color_counts, key=color_counts.get)
        confidence = color_counts[detected_color] / total

        return {
            "color": detected_color,
            "confidence": float(confidence)
        }

    def verify(self, image_path, box, claimed_color):

        result = self.estimate_color(image_path, box)

        detected_color = result["color"]
        confidence = result["confidence"]

        if detected_color == "unknown":
            status = "UNVERIFIED"

        elif confidence < 0.50:
            status = "UNVERIFIED"

        elif detected_color == claimed_color.lower():
            status = "SUPPORTED"

        else:
            status = "CONTRADICTED"

        return {
            "claimed_color": claimed_color.lower(),
            "detected_color": detected_color,
            "confidence": confidence,
            "status": status
        }


if __name__ == "__main__":

    verifier = ColorVerifier()

    box = [164.62, 62.41, 290.78, 193.05]

    # First estimate the actual color
    estimated = verifier.estimate_color(
        image_path="data/image.png",
        box=box
    )

    # Use the estimated color as the test claim
    result = verifier.verify(
        image_path="data/image.png",
        box=box,
        claimed_color=estimated["color"]
    )

    print("\nCOLOR VERIFIER")
    print("=" * 60)

    print("Claimed color:")
    print(result["claimed_color"])

    print("Detected color:")
    print(result["detected_color"])

    print("Confidence:")
    print(round(result["confidence"], 4))

    print("Status:")
    print(result["status"])