import torch
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration


MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

IMAGE_PATH = r"C:\Users\hp\vlm-hallucination-project\data\ChatGPT Image Sep 26, 2026, 12_41_30 PM.png"

QUESTION = (
    "Describe the image in detail. Mention the people, objects, "
    "colors, counts, and spatial relationships that you can observe."
)


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading model...")
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_NAME
).to(device)

print("Model loaded.")


image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": QUESTION
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

print("\nGenerating VLM response...\n")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=150
    )

generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("=" * 70)
print("VLM RESPONSE")
print("=" * 70)
print(generated_text)