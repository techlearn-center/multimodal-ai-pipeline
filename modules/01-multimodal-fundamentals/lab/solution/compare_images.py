"""Solution: Image Comparison Tool"""
import base64
import sys
from openai import OpenAI

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def compare_images(image1_path: str, image2_path: str) -> str:
    client = OpenAI()
    b64_1 = encode_image(image1_path)
    b64_2 = encode_image(image2_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images: similarities, differences, content, style, and overall assessment."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_2}"}},
            ],
        }],
        max_tokens=800,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_images.py <image1> <image2>")
        sys.exit(1)
    print(compare_images(sys.argv[1], sys.argv[2]))
