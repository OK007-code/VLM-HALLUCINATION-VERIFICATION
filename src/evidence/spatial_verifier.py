class SpatialVerifier:

    def get_center(self, box):

        x1, y1, x2, y2 = box

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return center_x, center_y

    def verify(self, relation, object1_detections, object2_detections):

        # No evidence for either object
        if not object1_detections or not object2_detections:

            return {
                "status": "UNVERIFIED",
                "relation": relation
            }

        # Use strongest detection for each object
        object1 = max(
            object1_detections,
            key=lambda x: x["confidence"]
        )

        object2 = max(
            object2_detections,
            key=lambda x: x["confidence"]
        )

        center1_x, center1_y = self.get_center(
            object1["box"]
        )

        center2_x, center2_y = self.get_center(
            object2["box"]
        )

        # ---------------------------------------------
        # Check spatial relationship
        # ---------------------------------------------

        if relation == "left":

            is_correct = center1_x < center2_x

        elif relation == "right":

            is_correct = center1_x > center2_x

        elif relation == "above":

            is_correct = center1_y < center2_y

        elif relation == "below":

            is_correct = center1_y > center2_y

        else:

            return {
                "status": "UNVERIFIED",
                "relation": relation
            }

        # ---------------------------------------------
        # Result
        # ---------------------------------------------

        if is_correct:

            status = "SUPPORTED"

        else:

            status = "CONTRADICTED"

        return {
            "status": status,
            "relation": relation,
            "object1_center": [
                center1_x,
                center1_y
            ],
            "object2_center": [
                center2_x,
                center2_y
            ]
        }


if __name__ == "__main__":

    verifier = SpatialVerifier()

    # Dog is on the left
    dog = [
        {
            "object": "dog",
            "confidence": 0.92,
            "box": [50, 100, 150, 250]
        }
    ]

    # Car is on the right
    car = [
        {
            "object": "car",
            "confidence": 0.89,
            "box": [300, 100, 500, 300]
        }
    ]

    result = verifier.verify(
        relation="left",
        object1_detections=dog,
        object2_detections=car
    )

    print("\nSPATIAL VERIFIER")
    print("=" * 60)

    print("Relation:", result["relation"])
    print("Dog center:", result["object1_center"])
    print("Car center:", result["object2_center"])
    print("Status:", result["status"])