#!/usr/bin/env python3
import os
import argparse
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class ConvHead(nn.Module):
    def __init__(self, in_ch, rep_size=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, rep_size)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.avg(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(checkpoint_path, device="cpu"):
    """
    Build the detector and load weights.
    Robust to odd device strings (' cpu ', 'CUDA:0') and state_dict formats.
    """
    device = torch.device(str(device).strip())

    # Base model
    det_wts = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    bb_wts = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=det_wts,
        weights_backbone=bb_wts,
    )

    # Compact head (2 classes: bg + flank)
    in_ch = model.backbone.out_channels  # 256
    model.roi_heads.box_head = ConvHead(in_ch, rep_size=1024)
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, num_classes=2)

    # Load fine-tuned weights with fallbacks
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            cleaned[k[7:]] = v if k.startswith("module.") else v
            if not k.startswith("module."):
                cleaned[k] = v
        state = cleaned

    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Detect & crop giraffe flanks (no JSON needed)")
    parser.add_argument("--images_dir", required=True, help="Folder with test images")
    parser.add_argument("--checkpoint", required=True, help="Path to your .pth checkpoint")
    parser.add_argument("--output_dir", default="crops", help="Where to save color crops")
    parser.add_argument("--score_thresh", type=float, default=0.5, help="Min detection score")
    parser.add_argument("--grayscale", action="store_true", help="Convert input to gray (×3 channels)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(args.checkpoint, device=device)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    img_paths = []
    for e in exts:
        img_paths += glob.glob(os.path.join(args.images_dir, e))

    for path in sorted(img_paths):
        fname = os.path.basename(path)
        orig_bgr = cv2.imread(path)
        if orig_bgr is None:
            print(f"⚠️  Could not read {fname}, skipping")
            continue

        if args.grayscale:
            gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
            img = np.stack([gray, gray, gray], axis=2)
        else:
            img = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255)
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)[0]

        boxes = outputs["boxes"].cpu().numpy() if "boxes" in outputs else np.zeros((0, 4))
        scores = outputs["scores"].cpu().numpy() if "scores" in outputs else np.zeros((0,))
        if boxes.size == 0 or scores.max(initial=0.0) < args.score_thresh:
            print(f"{fname}: no detection ≥ {args.score_thresh}")
            continue

        idx = int(np.argmax(scores))
        x1, y1, x2, y2 = boxes[idx].astype(int)

        crop = orig_bgr[y1:y2, x1:x2]
        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, crop)
        print(f"{fname}: saved crop → {out_path}")


if __name__ == "__main__":
    main()
