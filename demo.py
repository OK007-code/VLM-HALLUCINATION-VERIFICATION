import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration
)

from src.claim_extractor import extract_and_normalize_claims
from src.pipeline import VLMVerificationPipeline


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD INTERACTIVE VLM
# ============================================================

print("=" * 70)
print("Loading Interactive VLM")
print("=" * 70)

print(f"Model : {MODEL_NAME}")
print(f"Device: {device}")

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("Loading SmolVLM...")

model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

print("SmolVLM loaded successfully.")


# ============================================================
# GENERATE VLM RESPONSE
# ============================================================

def ask_vlm(image_path):

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": (
                        "Answer the user's question using complete sentences. "
                        "Describe every visible object, count, color, spatial "
                        "relationship, and readable text that is relevant. "
                        "Do not answer with only one word.\n\n"
                        "Describe the image and report only visible, verifiable facts."
                    )
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            min_new_tokens=20,
            do_sample=False
        )

    # Only take newly generated tokens.
    # This prevents "User: ..." and the prompt
    # from appearing in the final VLM response.

    input_length = inputs["input_ids"].shape[1]

    generated_only = generated_ids[:, input_length:]

    response = processor.batch_decode(
        generated_only,
        skip_special_tokens=True
    )[0]

    return response.strip()


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    image_path,
    vlm_answer,
    results
):

    print("\n")

    print("=" * 70)
    print("VLM HALLUCINATION VERIFICATION")
    print("=" * 70)

    print(f"\nImage: {image_path}")

    # --------------------------------------------------------
    # VLM RESPONSE
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("VLM RESPONSE")
    print("-" * 70)

    print(vlm_answer)

    # --------------------------------------------------------
    # ATOMIC CLAIMS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ATOMIC CLAIMS")
    print("=" * 70)

    claims = results.get("claims", [])

    if not claims:

        print("No verifiable claim was extracted.")

    else:

        for i, claim in enumerate(claims, 1):

            print(f"\nClaim {i}")

            print(f"  Type: {claim.get('type')}")

            print(f"  Data: {claim.get('data')}")

    # --------------------------------------------------------
    # CLAIM-WISE VERIFICATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLAIM-WISE VERIFICATION")
    print("=" * 70)

    verification_results = results.get(
        "results",
        []
    )

    if not verification_results:

        print("No verification results available.")

    else:

        for i, result in enumerate(
            verification_results,
            1
        ):

            print(f"\nClaim {i}")

            print(
                f"  Claim: "
                f"{result.get('claim')}"
            )

            # ------------------------------------------------
            # OBJECT
            # ------------------------------------------------

            if "evidence" in result:

                evidence = result["evidence"]

                print(
                    f"  Evidence Source: "
                    f"{evidence.get('source', 'N/A')}"
                )

                if "iou" in evidence:

                    print(
                        f"  IoU: "
                        f"{round(evidence.get('iou', 0), 4)}"
                    )

                if "gate" in result:

                    gate = result["gate"]

                    if isinstance(gate, dict):

                        print(
                            f"  Gate: "
                            f"{gate.get('state', 'N/A')}"
                        )

            # ------------------------------------------------
            # COUNT
            # ------------------------------------------------

            if "count_verification" in result:

                count_result = result[
                    "count_verification"
                ]

                print(
                    f"  Claimed Count: "
                    f"{count_result.get('claimed_count')}"
                )

                print(
                    f"  YOLO Count: "
                    f"{count_result.get('yolo_count')}"
                )

                print(
                    f"  DINO Count: "
                    f"{count_result.get('dino_count')}"
                )

                print(
                    f"  Detected Count: "
                    f"{count_result.get('detected_count')}"
                )

            # ------------------------------------------------
            # COLOR
            # ------------------------------------------------

            if "color_verification" in result:

                color_result = result[
                    "color_verification"
                ]

                print(
                    f"  Claimed Color: "
                    f"{color_result.get('claimed_color')}"
                )

                print(
                    f"  Detected Color: "
                    f"{color_result.get('detected_color')}"
                )

                print(
                    f"  Color Confidence: "
                    f"{round(color_result.get('confidence', 0), 4)}"
                )

            # ------------------------------------------------
            # SPATIAL
            # ------------------------------------------------

            if "spatial_verification" in result:

                spatial_result = result[
                    "spatial_verification"
                ]

                print(
                    f"  Spatial Result: "
                    f"{spatial_result}"
                )

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            if "ocr_verification" in result:

                ocr_result = result[
                    "ocr_verification"
                ]

                print(
                    f"  Claimed Text: "
                    f"{ocr_result.get('claimed_text')}"
                )

                print(
                    f"  Detected Text: "
                    f"{ocr_result.get('detected_text')}"
                )

                print(
                    f"  OCR Confidence: "
                    f"{round(ocr_result.get('confidence', 0), 4)}"
                )

            # ------------------------------------------------
            # FINAL VERDICT
            # ------------------------------------------------

            print(
                f"  Final Verdict: "
                f"{result.get('verdict', result.get('status', 'N/A'))}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_claims = len(
        verification_results
    )

    supported = sum(
        1
        for result in verification_results
        if result.get("verdict") == "SUPPORTED"
    )

    contradicted = sum(
        1
        for result in verification_results
        if result.get("verdict") == "CONTRADICTED"
    )

    unverified = sum(
        1
        for result in verification_results
        if result.get("verdict") == "UNVERIFIED"
    )

    print(f"Total Claims : {total_claims}")
    print(f"Supported    : {supported}")
    print(f"Contradicted : {contradicted}")
    print(f"Unverified   : {unverified}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)
    print("EVIDENCE-GROUNDED VLM HALLUCINATION VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_path = input(
        "\nEnter image path: "
    ).strip()

    if not image_path:

        print(
            "Error: Image path cannot be empty."
        )

        return

    # --------------------------------------------------------
    # STEP 1
    # GENERATE RESPONSE USING SMOLVLM
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 1: GENERATING VLM RESPONSE")
    print("-" * 70)

    vlm_answer = ask_vlm(
        image_path
    )

    print("\nVLM response generated successfully.")

    # --------------------------------------------------------
    # STEP 2: EXTRACT ATOMIC CLAIMS
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 2: EXTRACTING ATOMIC CLAIMS FROM VLM RESPONSE")
    print("-" * 70)

    extracted_claims = extract_and_normalize_claims(vlm_answer)

    if not extracted_claims:
        print("No atomic claims were extracted from the VLM response.")
    else:
        for index, claim in enumerate(extracted_claims, 1):
            print(f"Claim {index}: {claim}")

    # --------------------------------------------------------
    # STEP 2
    # LOAD EXISTING VERIFICATION PIPELINE
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 2: LOADING VERIFICATION PIPELINE")
    print("-" * 70)

    pipeline = VLMVerificationPipeline()

    # --------------------------------------------------------
    # STEP 3
    # VERIFY RAW VLM RESPONSE
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("STEP 3: RUNNING EVIDENCE-GROUNDED VERIFICATION")
    print("-" * 70)

    results = pipeline.verify(
        image_path=image_path,
        vlm_answer=vlm_answer,
        extracted_claims=extracted_claims
    )

    # --------------------------------------------------------
    # STEP 4
    # DISPLAY EVERYTHING
    # --------------------------------------------------------

    print_results(
        image_path=image_path,
        vlm_answer=vlm_answer,
        results=results
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()