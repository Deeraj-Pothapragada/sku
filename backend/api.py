from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from inference import SKURecognizer

MODEL_PATH = "full_arcface_model.pt"
YOLO_PATH  = "yolo_weights.pt"

ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["rec"] = SKURecognizer(MODEL_PATH, YOLO_PATH)
    yield
    ml.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten to your real frontend domain later
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ready" if "rec" in ml else "loading"}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 10_000_000:
        raise HTTPException(413, "Image too large (10MB max)")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")
    if "rec" not in ml:
        raise HTTPException(503, "Model still loading")
    return {"detections": ml["rec"].detect_and_recognize_pil(img)}
