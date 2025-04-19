
![14.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/xVgE0XmLAzc-7BPVXXDNZ.png)

# **Minc-Materials-23**

> **Minc-Materials-23** is a visual material classification model fine-tuned from **google/siglip2-base-patch16-224** using the **SiglipForImageClassification** architecture. It classifies images into 23 common material types based on visual features, ideal for applications in material recognition, construction, retail, robotics, and beyond.

```py
Classification Report:
               precision    recall  f1-score   support

        brick     0.8325    0.8278    0.8301      2125
       carpet     0.7318    0.7539    0.7427      2125
      ceramic     0.6484    0.6579    0.6531      2125
       fabric     0.6248    0.5666    0.5943      2125
      foliage     0.9102    0.9205    0.9153      2125
         food     0.8588    0.8899    0.8740      2125
        glass     0.7799    0.6753    0.7238      2125
         hair     0.9267    0.9520    0.9392      2125
      leather     0.7464    0.7826    0.7641      2125
        metal     0.6491    0.6626    0.6558      2125
       mirror     0.7668    0.6127    0.6811      2125
        other     0.8637    0.8198    0.8411      2125
      painted     0.6813    0.8391    0.7520      2125
        paper     0.7393    0.7261    0.7327      2125
      plastic     0.6142    0.5304    0.5692      2125
polishedstone     0.7435    0.7449    0.7442      2125
         skin     0.8995    0.9228    0.9110      2125
          sky     0.9584    0.9751    0.9666      2125
        stone     0.7567    0.7289    0.7426      2125
         tile     0.7108    0.6847    0.6975      2125
    wallpaper     0.7825    0.8193    0.8005      2125
        water     0.8993    0.8781    0.8886      2125
         wood     0.6281    0.7685    0.6912      2125

     accuracy                         0.7713     48875
    macro avg     0.7719    0.7713    0.7700     48875
 weighted avg     0.7719    0.7713    0.7700     48875
```

![download (2).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/QNv3UQpQg6RkPLXaUFE2g.png)

The model categorizes images into the following 23 classes:

- **0:** brick  
- **1:** carpet  
- **2:** ceramic  
- **3:** fabric  
- **4:** foliage  
- **5:** food  
- **6:** glass  
- **7:** hair  
- **8:** leather  
- **9:** metal  
- **10:** mirror  
- **11:** other  
- **12:** painted  
- **13:** paper  
- **14:** plastic  
- **15:** polishedstone  
- **16:** skin  
- **17:** sky  
- **18:** stone  
- **19:** tile  
- **20:** wallpaper  
- **21:** water  
- **22:** wood  

---

# **Run with Transformers 🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```

---

# **Intended Use**

**Minc-Materials-23** is tailored for:

- **Architecture & Construction:** Material identification from site photos or plans.  
- **Retail & Inventory:** Recognizing product materials in e-commerce.  
- **Robotics & AI Vision:** Enabling object material perception.  
- **Environmental Monitoring:** Detecting materials in natural vs. urban environments.   
- **Education & Research:** Teaching material properties and classification techniques.
