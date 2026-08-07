FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py api.py ./
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')"

COPY models/full_arcface_model.pt models/yolo_weights.pt ./models/

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
