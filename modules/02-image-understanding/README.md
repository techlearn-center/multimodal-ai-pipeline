# Module 02: Image Understanding

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner-Intermediate |
| **Prerequisites** | Module 01 completed, environment running |

---

## Learning Objectives
- Use vision models for image captioning, OCR, and visual QA
- Build an image classification pipeline
- Process batches of images efficiently
- Handle different image formats and sizes

---

## 1. Image Captioning

### Single Image
```python
from src.processors.image_processor import ImageProcessor

processor = ImageProcessor(model="gpt-4o")
caption = processor.caption("data/images/sample.jpg", detail="high")
print(caption)
```

### Batch Processing with ThreadPoolExecutor
```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def process_batch(image_dir: str, max_workers: int = 4):
    processor = ImageProcessor()
    images = list(Path(image_dir).glob("*.jpg"))
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(processor.caption, str(img)): img
            for img in images
        }
        for future in futures:
            img = futures[future]
            try:
                results[img.name] = future.result()
            except Exception as e:
                results[img.name] = f"Error: {e}"

    return results

results = process_batch("data/images/")
for name, caption in results.items():
    print(f"{name}: {caption[:100]}...")
```

---

## 2. OCR with Vision Models

### Extracting Text from Documents
```python
# Vision model OCR (handles complex layouts, handwriting, tables)
text = processor.extract_text("data/documents/receipt.jpg")
print(text)

# Structured extraction from invoices
structured = processor.analyze(
    "data/documents/invoice.jpg",
    "Extract as JSON: company_name, invoice_number, date, total_amount, line_items"
)
print(structured)
```

### Comparing with Traditional OCR (Tesseract)
```python
import pytesseract
from PIL import Image

# Traditional OCR
img = Image.open("data/documents/receipt.jpg")
tesseract_text = pytesseract.image_to_string(img)
print("Tesseract:", tesseract_text)

# Vision model OCR
vision_text = processor.extract_text("data/documents/receipt.jpg")
print("Vision:", vision_text)

# Vision models win on: handwriting, complex layouts, tables, mixed languages
# Tesseract wins on: speed, cost, offline processing
```

---

## 3. Visual Question Answering

```python
# Ask specific questions about charts
answer = processor.analyze(
    "data/images/chart.png",
    "What was the revenue in Q3? Give the exact number."
)
print(answer)

# Multi-image comparison
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these two product photos. Which has better lighting?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_1}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_2}"}},
        ],
    }],
)
```

---

## 4. Hands-On Lab

Build an image analysis API endpoint that:
1. Accepts image uploads via POST
2. Returns caption, detected text, and answers to custom questions
3. Supports batch processing of multiple images

```bash
cd modules/02-image-understanding/lab/starter/
```

## Validation
```bash
bash modules/02-image-understanding/validation/validate.sh
```

---

**Next: [Module 03 - Document Processing →](../03-document-processing/)**
