import json
import re


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

COLORS = (
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "brown", "black", "white", "gray", "grey", "silver", "gold",
)

SPATIAL_RELATIONS = {
    "to the left of": "left",
    "on the left of": "left",
    "left of": "left",
    "to the right of": "right",
    "on the right of": "right",
    "right of": "right",
    "above": "above",
    "below": "below",
    "under": "below",
    "behind": "behind",
    "in front of": "in_front_of",
}


# ============================================================
# PROMPT FOR AUTOMATIC ATOMIC CLAIM EXTRACTION
# ============================================================

EXTRACTION_PROMPT = """
You are an atomic claim extraction module.

Given the following Vision-Language Model response, extract ALL
independent factual claims that can be verified against an image.

Do NOT use a predefined object list.
Do NOT assume specific objects.
Extract whatever objects, counts, colors, spatial relations,
or visible text are actually mentioned in the response.

Break compound sentences into separate atomic claims.

Allowed claim types:

1. object
   Example:
   "There is a cat."
   ->
   {
       "type": "object",
       "data": {
           "object": "cat"
       }
   }

2. count
   Example:
   "There are three people."
   ->
   {
       "type": "count",
       "data": {
           "object": "person",
           "count": 3
       }
   }

3. color
   Example:
   "The car is red."
   ->
   {
       "type": "color",
       "data": {
           "object": "car",
           "color": "red"
       }
   }

4. spatial
   Example:
   "The cat is to the left of the vase."
   ->
   {
       "type": "spatial",
       "data": {
           "object1": "cat",
           "relation": "left",
           "object2": "vase"
       }
   }

5. ocr
   Example:
   "The sign says STOP."
   ->
   {
       "type": "ocr",
       "data": {
           "text": "STOP"
       }
   }

Only extract claims that describe observable visual facts.

Do NOT extract:
- opinions
- guesses
- uncertainty
- explanations
- general knowledge
- subjective descriptions
- facts that cannot reasonably be checked from the image

Return ONLY valid JSON.

The JSON must be:

{
    "claims": [
        {
            "type": "...",
            "data": {...}
        }
    ]
}

VLM RESPONSE:
"""


# ============================================================
# CLEAN MODEL OUTPUT
# ============================================================

def _clean_json(text):

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    text = text.strip()

    # Find JSON object if model added extra text
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text.strip()


# ============================================================
# PARSE JSON
# ============================================================

def parse_claim_json(response):

    if not response:
        return []

    cleaned = _clean_json(response)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        claims = parsed.get("claims", [])
    elif isinstance(parsed, list):
        claims = parsed
    else:
        return []

    if not isinstance(claims, list):
        return []

    valid_claims = []

    for claim in claims:

        if not isinstance(claim, dict):
            continue

        claim_type = claim.get("type")
        data = claim.get("data")

        if not claim_type or not isinstance(data, dict):
            continue

        if claim_type not in {
            "object",
            "count",
            "color",
            "spatial",
            "ocr"
        }:
            continue

        valid_claims.append({
            "type": claim_type,
            "data": data
        })

    return valid_claims


# ============================================================
# NORMALIZE CLAIM
# ============================================================

def normalize_claim(claim):

    claim_type = claim.get("type")
    data = claim.get("data", {})

    if claim_type == "object":

        if "object" not in data:
            return None

        data["object"] = str(data["object"]).strip().lower()

    elif claim_type == "count":

        if "object" not in data or "count" not in data:
            return None

        data["object"] = str(data["object"]).strip().lower()

        try:
            data["count"] = int(data["count"])
        except (ValueError, TypeError):
            return None

    elif claim_type == "color":

        if "object" not in data or "color" not in data:
            return None

        data["object"] = str(data["object"]).strip().lower()
        data["color"] = str(data["color"]).strip().lower()

    elif claim_type == "spatial":

        if (
            "object1" not in data
            or "relation" not in data
            or "object2" not in data
        ):
            return None

        data["object1"] = str(data["object1"]).strip().lower()
        data["object2"] = str(data["object2"]).strip().lower()
        data["relation"] = str(data["relation"]).strip().lower()

    elif claim_type == "ocr":

        if "text" not in data:
            return None

        data["text"] = str(data["text"]).strip()

    else:
        return None

    return {
        "type": claim_type,
        "data": data
    }


# ============================================================
# DEDUPLICATE CLAIMS
# ============================================================

def deduplicate_claims(claims):

    unique_claims = []
    seen = set()

    for claim in claims:

        try:
            key = json.dumps(
                claim,
                sort_keys=True,
                ensure_ascii=False
            )
        except TypeError:
            continue

        if key in seen:
            continue

        seen.add(key)
        unique_claims.append(claim)

    return unique_claims


def _singularize(noun):

    noun = noun.strip(" ,.;:!?\"'").lower()

    if noun == "people":
        return "person"

    if noun.endswith("ies"):
        return noun[:-3] + "y"

    if noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]

    return noun


def _number_value(value):
    value = value.lower()
    return NUMBER_WORDS.get(value, int(value) if value.isdigit() else None)


def extract_claims_from_text(vlm_answer):
    """Extract verifiable claims directly when no extraction model is supplied."""

    text = re.sub(r"\s+", " ", str(vlm_answer)).strip()
    claims = []

    # Text on signs/labels is a particularly reliable, explicit claim form.
    for match in re.finditer(
        r"\b(?:says|reads|reading)\s+['\"]?([^'\".!?]+)",
        text,
        flags=re.IGNORECASE,
    ):
        claims.append({"type": "ocr", "data": {"text": match.group(1).strip()}})

    # Spatial claims are extracted before object phrases so their two entities
    # remain separate atomic claims.
    relation_pattern = "|".join(
        sorted(SPATIAL_RELATIONS, key=len, reverse=True)
    )
    for match in re.finditer(
        rf"\bthe\s+([a-z][a-z -]*?)\s+(?:is\s+)?({relation_pattern})\s+the\s+([a-z][a-z -]*?)(?=[,.!?;]|\s+and\s+|$)",
        text,
        flags=re.IGNORECASE,
    ):
        claims.append({
            "type": "spatial",
            "data": {
                "object1": _singularize(match.group(1)),
                "relation": SPATIAL_RELATIONS[match.group(2).lower()],
                "object2": _singularize(match.group(3)),
            },
        })

    # Counts support both digits and common number words.
    number_pattern = "|".join(NUMBER_WORDS)
    for match in re.finditer(
        rf"\b(?:there (?:are|is)|(?:we|i) can see|the image shows)\s+"
        rf"({number_pattern}|\d+)\s+([a-z][a-z -]*?)(?=[,.!?;]|\s+(?:and|near|beside|behind|under|on|in)\s+|$)",
        text,
        flags=re.IGNORECASE,
    ):
        count = _number_value(match.group(1))
        if count is not None:
            claims.append({
                "type": "count",
                "data": {"object": _singularize(match.group(2)), "count": count},
            })

    # Adjective-noun and noun-is-adjective color descriptions.
    color_pattern = "|".join(COLORS)
    for match in re.finditer(
        rf"\b({color_pattern})\s+([a-z][a-z -]*?)(?=[,.!?;]|\s+and\s+|$)",
        text,
        flags=re.IGNORECASE,
    ):
        claims.append({
            "type": "color",
            "data": {
                "object": _singularize(match.group(2)),
                "color": match.group(1).lower(),
            },
        })

    for match in re.finditer(
        rf"\bthe\s+([a-z][a-z -]*?)\s+is\s+(?:colored\s+)?({color_pattern})\b",
        text,
        flags=re.IGNORECASE,
    ):
        claims.append({
            "type": "color",
            "data": {
                "object": _singularize(match.group(1)),
                "color": match.group(2).lower(),
            },
        })

    # Basic existence statements cover the common VLM forms without a fixed
    # object vocabulary.
    for match in re.finditer(
        r"\b(?:there is|there's|i can see|visible is|the image shows)\s+"
        r"(?:a|an|the)?\s*([a-z][a-z -]*?)(?=[,.!?;]|\s+and\s+|$)",
        text,
        flags=re.IGNORECASE,
    ):
        noun = _singularize(match.group(1))
        if noun and not noun.isdigit():
            claims.append({"type": "object", "data": {"object": noun}})

    normalized_claims = []
    for claim in claims:
        normalized = normalize_claim(claim)
        if normalized is not None:
            normalized_claims.append(normalized)

    return deduplicate_claims(normalized_claims)


# ============================================================
# MAIN CLAIM EXTRACTION FUNCTION
# ============================================================

def extract_and_normalize_claims(
    vlm_answer,
    extractor_model=None,
    extractor_processor=None,
    device="cpu"
):

    if not vlm_answer:
        return []

    if extractor_model is None or extractor_processor is None:
        return extract_claims_from_text(vlm_answer)

    # --------------------------------------------------------
    # Build extraction prompt
    # --------------------------------------------------------

    prompt = (
        EXTRACTION_PROMPT
        + "\n"
        + str(vlm_answer)
        + "\n\nReturn ONLY JSON."
    )

    # --------------------------------------------------------
    # Generate JSON using the supplied VLM/text model
    # --------------------------------------------------------

    try:

        inputs = extractor_processor(
            text=prompt,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        output = extractor_model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False
        )

        response = extractor_processor.decode(
            output[0],
            skip_special_tokens=True
        )

    except Exception as e:

        print("Claim extraction failed:")
        print(e)

        return []

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    claims = parse_claim_json(
        response
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized_claims = []

    for claim in claims:

        normalized = normalize_claim(
            claim
        )

        if normalized is not None:
            normalized_claims.append(
                normalized
            )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    normalized_claims = deduplicate_claims(
        normalized_claims
    )

    return normalized_claims