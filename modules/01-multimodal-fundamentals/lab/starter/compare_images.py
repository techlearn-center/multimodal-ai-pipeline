"""
Lab: Image Comparison Tool
Complete the TODO sections.
"""
import base64
import sys
from openai import OpenAI

def encode_image(path: str) -> str:
    """TODO: Read the image file and return base64-encoded string."""
    pass

def compare_images(image1_path: str, image2_path: str) -> str:
    """TODO: Send both images to GPT-4o and get a structured comparison."""
    pass

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_images.py <image1> <image2>")
        sys.exit(1)
    print(compare_images(sys.argv[1], sys.argv[2]))
