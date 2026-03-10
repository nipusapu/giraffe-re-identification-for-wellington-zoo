#!/usr/bin/env python3
import os
import argparse
import glob

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(checkpoint_path, device="cpu"):
    """
    Build the detector and load weights.
    Robust to odd device strings (' cpu ', 'CUDA:0') and state_dict formats.
    """
    device = torch.device(str(device).strip())

    # Base model
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    # Load fine-tuned weights with fallbacks
    state = torch.load(checkpoint_path, map_location="cpu")

    # Handle the save format
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            cleaned[k[7:]] = v if k.startswith("module.") else v
            if not k.startswith("module."):
                cleaned[k] = v
        state = cleaned

    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print("Missing keys:", len(missing), "Unexpected:", len(unexpected), flush=True)
    if missing: print("Missing sample:", missing[:10], flush=True)
    if unexpected: print("Unexpected sample:", unexpected[:10], flush=True)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Detect & crop giraffe flanks (no JSON needed)")
    parser.add_argument("--images_dir", required=True, help="Folder with test images")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pth checkpoint")
    parser.add_argument("--output_dir", default="crops", help="Where to save color crops")
    parser.add_argument("--score_thresh", type=float, default=0.5, help="Min detection score")
    parser.add_argument("--grayscale", action="store_true", help="Convert input to gray (x3 channels)")
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
        # tensor = .float().div(255)
        tensor = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        tensor = F.normalize(tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

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
