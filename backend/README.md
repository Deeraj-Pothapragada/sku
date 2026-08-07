# SKU Recognition — Deployment Bundle v1

## Contents
- `full_arcface_model.pt` — model checkpoint (backbone + head + prototypes + SKU list)
- `prototypes.pt` — standalone 11682x256 prototype matrix (redundant; also inside checkpoint)
- `skus.json` — ordered SKU label list (index i <-> prototype row i)
- `yolo_weights.pt` — object detector weights
- `inference.py` — self-contained inference module (`SKURecognizer` class)
- `requirements.txt`

## Config (MUST match at inference — mismatches fail silently)
- Backbone: `dinov2_vitb14_reg`
- Input size: 336x336 (multiple of 14)
- Embedding dim: 256
- Loss trained with: arcface
- Unfrozen backbone blocks: last 2 + final norm

## IMPORTANT: DINOv2 backbone loading
`inference.py` calls `torch.hub.load("facebookresearch/dinov2", ...)`, which
downloads stock DINOv2 weights from the internet, then overlays our fine-tuned
`backbone_state`. In Docker EITHER:
  (a) allow network access on first run, OR
  (b) pre-download the hub cache and bake it into the image (recommended).
The fine-tuned weights are already in the checkpoint — the hub download only
provides the architecture + unmodified layers.

## Preprocessing (exact)
Resize to 336x336, ToTensor,
Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
Encoded in `inference.py` — do not re-derive.

## Classification method
Nearest-PROTOTYPE (one vector per SKU, cosine similarity, argmax).
Chosen over full-gallery kNN: ~11682 comparisons vs ~hundreds of thousands,
with near-identical accuracy after ArcFace training.

## Unknown / no-match handling
Top prototype cosine similarity below UNKNOWN_THRESHOLD (currently 0.35 in
inference.py) -> flagged "UNKNOWN". THIS THRESHOLD IS A PLACEHOLDER — calibrate
on real deployment photos before trusting it.

## Expected accuracy (validation, held-out families)
- recall@1: 0.7195 | recall@3: 0.8112 | recall@5: 0.8179
- nearest-prototype recall@1: 0.7111
- NOTE: measured on clean catalog images. Real-world photos (lighting,
  occlusion, angle) will differ. recall@5 >> recall@1, so a top-5 shortlist
  (human review or OCR rerank) is more reliable than trusting top-1.

## Known limitations
- Flavor variants with near-identical packaging: often confused (label text
  not resolved at this resolution).
- Same-flavor-different-size SKUs: NOT visually separable from isolated photos;
  disambiguate via metadata/weight/barcode downstream.
- Possible residual label noise in training data.
