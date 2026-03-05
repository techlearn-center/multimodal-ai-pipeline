# Module 07: Vision-Language Models

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 06 completed |

---

## Learning Objectives
- Understand CLIP, LLaVA, and open-source VLMs
- Use CLIP embeddings for image search
- Compare commercial vs open-source options

---

## 1. CLIP Embeddings

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image-text similarity scoring
image = Image.open("data/images/cat.jpg")
inputs = processor(
    text=["a photo of a cat", "a photo of a dog", "a photo of a car"],
    images=image,
    return_tensors="pt",
    padding=True,
)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)
print(f"Cat: {probs[0][0]:.2%}, Dog: {probs[0][1]:.2%}, Car: {probs[0][2]:.2%}")
```

### Building an Image Search Engine
```python
import torch

def build_image_index(image_dir: str):
    images = list(Path(image_dir).glob("*.jpg"))
    embeddings = []
    for img_path in images:
        image = Image.open(img_path)
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        embeddings.append(emb.squeeze().numpy())
    return images, embeddings

def search_images(query: str, images, embeddings, top_k=5):
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    similarities = [torch.cosine_similarity(text_emb, torch.tensor(e).unsqueeze(0)).item() for e in embeddings]
    ranked = sorted(zip(images, similarities), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

---

## 2. LLaVA (Open-Source Vision-Language Model)

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

image = Image.open("data/images/diagram.png")
prompt = "USER: <image>\nDescribe this diagram in detail.\nASSISTANT:"
inputs = processor(prompt, image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## 3. Comparison

| Feature | GPT-4o | Gemini Pro | LLaVA 1.5 | CLIP |
|---------|--------|-----------|-----------|------|
| Hosted | Yes | Yes | Self-hosted | Self-hosted |
| Cost | Per token | Per token | GPU cost | GPU cost |
| Quality | Highest | High | Good | Embeddings only |
| Latency | ~2-5s | ~1-3s | ~5-15s | ~0.1s |
| Privacy | Data sent to API | Data sent to API | Full privacy | Full privacy |

---

## 4. Hands-On Lab

Build a hybrid image search: CLIP for fast retrieval, GPT-4o for detailed analysis of top results.

## Validation
```bash
bash modules/07-vision-language-models/validation/validate.sh
```

**Next: [Module 08 →](../08-pipeline-orchestration/)**
