"""
Image Understanding Pipeline
Processes images using vision models for captioning, OCR, and analysis.
"""
import base64
from pathlib import Path

from openai import OpenAI
from PIL import Image


class ImageProcessor:
    """Process images using multimodal LLMs."""

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def resize_if_needed(self, image_path: str, max_size: int = 2048) -> str:
        img = Image.open(image_path)
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            resized_path = str(Path(image_path).with_suffix(".resized.jpg"))
            img.save(resized_path, "JPEG", quality=85)
            return resized_path
        return image_path

    def caption(self, image_path: str, detail: str = "auto") -> str:
        image_path = self.resize_if_needed(image_path)
        b64 = self.encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail. Include objects, colors, spatial relationships, and notable features."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}},
                ],
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content

    def extract_text(self, image_path: str) -> str:
        b64 = self.encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract ALL text visible in this image. Preserve formatting and layout."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def analyze(self, image_path: str, question: str) -> str:
        b64 = self.encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    import sys
    processor = ImageProcessor()
    if len(sys.argv) < 2:
        print("Usage: python image_processor.py <image_path> [question]")
        sys.exit(1)
    if len(sys.argv) > 2:
        print(processor.analyze(sys.argv[1], sys.argv[2]))
    else:
        print("=== Caption ===")
        print(processor.caption(sys.argv[1]))
        print("\n=== Extracted Text ===")
        print(processor.extract_text(sys.argv[1]))
