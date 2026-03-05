FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y ffmpeg tesseract-ocr poppler-utils libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8888
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
