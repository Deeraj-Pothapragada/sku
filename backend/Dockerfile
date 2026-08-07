FROM python:3.12-slim

WORKDIR /app

# system deps needed by opencv/ultralytics/pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bake the DINOv2 hub cache into the image at build time —
# avoids a network dependency on torch.hub at runtime
COPY inference.py api.py ./
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')"

COPY full_arcface_model.pt yolo_weights.pt ./

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
