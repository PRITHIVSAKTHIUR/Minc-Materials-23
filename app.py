import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Minc-Materials-23"  # Replace with your actual model path
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    0: "brick",
    1: "carpet",
    2: "ceramic",
    3: "fabric",
    4: "foliage",
    5: "food",
    6: "glass",
    7: "hair",
    8: "leather",
    9: "metal",
    10: "mirror",
    11: "other",
    12: "painted",
    13: "paper",
    14: "plastic",
    15: "polishedstone",
    16: "skin",
    17: "sky",
    18: "stone",
    19: "tile",
    20: "wallpaper",
    21: "water",
    22: "wood"
}

def classify_material(image):
    """Predicts the material type present in the uploaded image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {id2label[i]: round(probs[i], 3) for i in range(len(probs))}
    return predictions

# Gradio interface
iface = gr.Interface(
    fn=classify_material,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Material Prediction Scores"),
    title="Minc-Materials-23",
    description="Upload an image to identify the material type (e.g., brick, wood, plastic, metal, etc.)."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
