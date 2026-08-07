import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, d_in, d_out, d_hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, d_hidden),
            nn.GELU(), nn.Linear(d_hidden, d_out))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class SKURecognizer:
    def __init__(self, model_path, yolo_path, unknown_threshold=0.35):
        ckpt = torch.load(model_path, map_location=DEVICE)
        cfg = ckpt['config']

        self.img_size = cfg['img_size']
        self.unknown_threshold = unknown_threshold
        self.tfm = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.skus   = ckpt['skus']
        self.protos = ckpt['protos'].to(DEVICE)

        self.backbone = torch.hub.load('facebookresearch/dinov2', cfg['model']).to(DEVICE).eval()
        self.backbone.load_state_dict(ckpt['backbone_state'])

        d_in = ckpt['head_state']['net.0.weight'].shape[0]   # inferred from checkpoint, not guessed
        self.head = Head(d_in, cfg['emb_dim']).to(DEVICE).eval()
        self.head.load_state_dict(ckpt['head_state'])

        self.detector = YOLO(yolo_path)

    @torch.no_grad()
    def recognize_crop(self, pil_crop, topk=5):
        x = self.tfm(pil_crop.convert('RGB'))[None].to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=(DEVICE == 'cuda')):
            z = self.head(self.backbone(x).float())
        sims = (z @ self.protos.T).squeeze(0)
        vals, ids = sims.topk(min(topk, len(self.skus)))
        preds = [(self.skus[i], round(float(v), 4)) for i, v in zip(ids.tolist(), vals.tolist())]
        is_unknown = preds[0][1] < self.unknown_threshold
        return preds, is_unknown

    def detect_and_recognize_pil(self, img, det_conf=0.25, pad_frac=0.05, topk=5):
        img = img.convert('RGB')
        W, H = img.size
        results = self.detector.predict(img, conf=det_conf, verbose=False)[0]
        out = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dx, dy = int((x2 - x1) * pad_frac), int((y2 - y1) * pad_frac)
            x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
            x2, y2 = min(W, x2 + dx), min(H, y2 + dy)
            crop = img.crop((x1, y1, x2, y2))
            preds, is_unknown = self.recognize_crop(crop, topk=topk)
            out.append({
                'bbox': [x1, y1, x2, y2],
                'yolo_conf': float(box.conf[0]),
                'predictions': preds,
                'is_unknown': is_unknown,
            })
        return out
