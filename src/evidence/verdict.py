from src.config import VERDICTS


class VerdictEngine:

    def get_verdict(self, claim, gate_result):

        claim_type = claim["type"]
        gate = gate_result["gate"]

        # Currently handling object claims
        if claim_type == "object":

            # Reliable visual evidence supports
            # the existence claim.
            if gate in [
                "VERIFIABLE",
                "VERIFIABLE_WITH_SINGLE_SOURCE"
            ]:
                return "SUPPORTED"

            # No reliable evidence means
            # we cannot confirm or contradict the claim.
            if gate == "AMBIGUOUS":
                return "UNVERIFIED"

        return "UNVERIFIED"


if __name__ == "__main__":

    engine = VerdictEngine()

    test_claim = {
        "type": "object",
        "data": {
            "object": "cat"
        }
    }

    test_gate = {
        "gate": "VERIFIABLE",
        "source": "BOTH",
        "iou": 0.8449
    }

    verdict = engine.get_verdict(
        test_claim,
        test_gate
    )

    print("\nVERDICT ENGINE")
    print("=" * 60)
    print("Verdict:", verdict)