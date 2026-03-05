# Module 01: Multimodal AI Fundamentals

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner |
| **Prerequisites** | Python 3.10+, Docker, OpenAI API key |

---

## Learning Objectives

By the end of this module, you will:
- Understand what multimodal AI is and why it matters
- Know key architectures (CLIP, LLaVA, GPT-4V, Gemini)
- Set up your development environment with Docker
- Make your first multimodal API call

---

## 1. What is Multimodal AI?

Multimodal AI processes **multiple types of input** (text, images, audio, video) in a single model. Unlike traditional AI that handles one modality at a time, multimodal models understand relationships across modalities.

### Key Architectures

| Model | Modalities | Approach | When to Use |
|-------|-----------|----------|-------------|
| CLIP (OpenAI) | Image + Text | Contrastive learning, shared embedding space | Image search, zero-shot classification |
| LLaVA | Image + Text | Vision encoder + LLM decoder | Self-hosted visual QA |
| GPT-4V / GPT-4o | Image + Text + Audio | Native multimodal transformer | Highest quality, commercial use |
| Gemini | Image + Text + Audio + Video | Native multimodal | Video understanding, long context |
| Whisper | Audio to Text | Encoder-decoder for speech | Transcription, translation |

### Why Multimodal Matters for Production
- **Document understanding**: Process invoices, contracts, receipts automatically
- **Visual QA**: Answer questions about images and diagrams
- **Video analysis**: Summarize, search, and understand video content
- **Accessibility**: Audio descriptions, transcription, translation

---

## 2. Environment Setup

### Step 1: Clone and configure
```bash
cd multimodal-ai-pipeline
cp .env.example .env
# Edit .env with your actual API keys
nano .env
```

### Step 2: Start all services
```bash
docker compose up -d
```

### Step 3: Verify services
```bash
# Check API
curl http://localhost:8000/health
# Expected: {"status":"healthy","service":"multimodal-pipeline"}

# Check ChromaDB
curl http://localhost:8001/api/v1/heartbeat
# Expected: {"nanosecond heartbeat": ...}
```

### Step 4: Test the image processor
```bash
# Download a test image
curl -o data/images/test.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"

# Run the processor
docker compose exec app python -m src.processors.image_processor data/images/test.jpg
```

**Expected Output:**
```
=== Caption ===
The image shows a close-up photograph of an ant...

=== Extracted Text ===
No visible text found in this image.
```

---

## 3. Your First Multimodal API Call

### Using the OpenAI Vision API

```python
from openai import OpenAI
import base64

client = OpenAI()

# Method 1: URL-based image
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
                },
            },
        ],
    }],
    max_tokens=300,
)
print(response.choices[0].message.content)

# Method 2: Base64-encoded local image
with open("data/images/test.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }],
)
print(response.choices[0].message.content)
```

### Image Detail Levels

| Detail Level | Tokens Used | Best For |
|-------------|-------------|----------|
| `low` | 85 tokens | Quick classification, yes/no questions |
| `high` | 85-1105 tokens | OCR, detailed analysis, small text |
| `auto` | Varies | Let the model decide (recommended default) |

---

## 4. Hands-On Lab

### Task: Build an Image Comparison Tool

Create a script that takes two images and produces a structured comparison.

```bash
cd modules/01-multimodal-fundamentals/lab/starter/
# Complete the TODO sections in compare_images.py
```

Your tool should:
1. Load and encode both images to base64
2. Send them to GPT-4o in a single request
3. Return a structured comparison covering similarities, differences, content, and style

### Validation
```bash
bash modules/01-multimodal-fundamentals/validation/validate.sh
```

---

## Self-Check Questions

1. What is the difference between CLIP and GPT-4o for image understanding?
2. When would you use base64 encoding vs URL for images?
3. How do detail levels affect cost and quality?
4. What are the token costs of including images in API calls?

---

**Next: [Module 02 - Image Understanding →](../02-image-understanding/)**
