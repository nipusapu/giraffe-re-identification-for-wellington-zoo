#!/usr/bin/env python3
"""
MobileNetV3-Large + Faster R-CNN (FPN) fine-tuning for single-class detection
— grayscale pipeline, W&B logging, cosine schedule with warmup, and periodic checkpoints.

What this script does
---------------------
1) Loads train/val datasets from COCO JSONs + an images directory.
2) Converts every image to **grayscale** (replicated to 3 channels), applies CLAHE +
   sharpening, and performs light augmentations (multi-scale, CutOut, blur, brightness/contrast).
3) Initializes a torchvision **Faster R-CNN (MobileNetV3 Large + FPN)** detector:
   - Starts from COCO/Imagenet weights
   - Replaces ROI head with a small ConvHead → 1024-d → new classifier/regressor
4) Freezes most of the backbone, then **selectively unfreezes** the last N stages.
5) Trains with **AdamW** + **cosine decay** LR scheduler (5% warmup), logs to **Weights & Biases**.
6) Saves model checkpoints every 5 epochs and at the final epoch.

Inputs you must provide
-----------------------
• --images_dir  : folder where all image files live
• --train_json  : COCO-style JSON for the training split
• --val_json    : COCO-style JSON for the validation split

Key behaviors / assumptions
---------------------------
• Single class detection (labels are all ones); `num_classes=2` (background + 1 foreground).
• Images are processed as **grayscale** (3-channel replicated) but normalized with ImageNet stats.
• Validation uses `model.train()` to compute loss dictionaries (no gradients), so we can
  report comparable loss values; this is intentional.
• Loss = cls_loss + λ * box_reg_loss (`λ_loc=2.0`) to balance classification and localization.

Quick start
-----------
python trinscript.py \
  --images_dir data/images \
  --train_json data/annotations_train.json \
  --val_json   data/annotations_val.json \
  --output_dir checkpoints \
  --epochs 50 --batch_size 2 --lr 1e-4 --weight_decay 1e-4 \
  --project giraffe-reid

Outputs
-------
• Console logs with train/val losses per epoch
• W&B metrics: train_loss, val_loss, epoch
• Periodic checkpoints in --output_dir: model_epoch{N}.pth

Requirements
------------
pip install torch torchvision pycocotools opencv-python numpy wandb
"""

import os
import argparse
import random
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights
)
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from pycocotools.coco import COCO


def parse_args():
    """
    CLI with explicit, student-friendly help text.
    """
    p = argparse.ArgumentParser(
        description="Fine-tune Faster R-CNN (MobileNetV3-Large+FPN) with grayscale input and W&B logging."
    )
    p.add_argument("--images_dir",   required=True, help="Directory containing the images referenced by both JSONs.")
    p.add_argument("--train_json",   required=True, help="COCO JSON for TRAIN split (file_name must be relative to --images_dir).")
    p.add_argument("--val_json",     required=True, help="COCO JSON for VAL split (same file_name convention).")
    p.add_argument("--output_dir",   default="./checkpoints", help="Where to save model_epoch*.pth checkpoints.")
    p.add_argument("--epochs",       type=int,   default=50, help="Number of training epochs.")
    p.add_argument("--batch_size",   type=int,   default=2,  help="Batch size (detectors are memory-heavy; 2 is a safe default).")
    p.add_argument("--lr",           type=float, default=1e-4, help="Initial learning rate for AdamW.")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    p.add_argument("--project",      type=str,   default="giraffe-reid", help="Weights & Biases project name.")
    return p.parse_args()


def collate_fn(batch):
    """
    DataLoader collate function for detection models.
    Converts a list of (image_tensor, target_dict) pairs into:
      imgs:   list[Tensor] with shape (3, H, W)
      targets:list[dict]   with keys: boxes (FloatTensor[N,4]), labels (Int64Tensor[N]), image_id (Int64Tensor[1])
    """
    return tuple(zip(*batch))


class ConvHead(nn.Module):
    """
    Lightweight ROI feature head:
      conv3x3 → ReLU → conv3x3 → ReLU → global avg-pool → Linear to rep_size.
    Replaces the default box_head for simplicity and control over dimensionality.
    """
    def __init__(self, in_ch, rep_size=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.avg   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(in_ch, rep_size)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.avg(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def unfreeze_backbone(model, num_layers=3):
    """
    Unfreeze the last `num_layers` modules of the backbone body for fine-tuning.
    model.backbone.body is an IntermediateLayerGetter producing a sequence of modules.
    """
    body = model.backbone.body
    modules = list(body.children())
    for m in modules[-num_layers:]:
        for p in m.parameters():
            p.requires_grad = True


def get_model(num_classes=2):
    """
    Build the detector:
      • Load COCO detection weights and ImageNet backbone weights
      • Replace ROI box_head with our ConvHead(->1024)
      • Replace predictor with 2-class head (background + foreground)
    """
    det_wts = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    bb_wts  = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=det_wts, weights_backbone=bb_wts
    )
    in_ch = model.backbone.out_channels  # typically 256 from the FPN
    model.roi_heads.box_head = ConvHead(in_ch, rep_size=1024)
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, num_classes)
    return model


class CocoFlankDataset(Dataset):
    """
    COCO-style dataset (single class) with a fixed grayscale preprocessing pipeline:
      • BGR → GRAY → CLAHE → sharpen → replicate to 3 channels
      • Multi-scale resize to a random short side (600/800/1000)
      • Optional CutOut and Gaussian blur
      • Brightness and contrast jitter
      • ImageNet normalization (mean/std)
    """
    def __init__(self, coco_json, images_dir):
        self.coco       = COCO(coco_json)
        self.images_dir = images_dir
        self.ids        = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # ---- Load image ----
        img_id = self.ids[idx]
        info   = self.coco.loadImgs(img_id)[0]
        path   = os.path.join(self.images_dir, info["file_name"])
        bgr    = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)

        # ---- Grayscale + CLAHE + sharpen → replicate to 3 channels ----
        gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray    = clahe.apply(gray)
        sharp_k = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        gray    = cv2.filter2D(gray, -1, sharp_k)
        img     = np.stack([gray, gray, gray], axis=2)  # (H, W, 3)

        # to torch CHW in [0,1]
        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255)
        H0, W0 = img_t.shape[1:]

        # ---- Load GT boxes (COCO bbox is [x, y, w, h] → convert to [x1, y1, x2, y2]) ----
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)
        boxes   = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
        boxes  = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.size(0),), dtype=torch.int64)  # single foreground class

        # ---- Multi-scale resize (random short-side) ----
        tgt = random.choice([600, 800, 1000])
        if H0 < W0:
            new_H, new_W = tgt, int(W0 * tgt / H0)
        else:
            new_W, new_H = tgt, int(H0 * tgt / W0)
        scale = new_H / H0  # uniform scaling (same for x/y)
        img_t = F.resize(img_t, [new_H, new_W])
        boxes = boxes * scale

        # ---- CutOut (random black rectangle) ----
        if random.random() < 0.5:
            _, H, W = img_t.shape
            cw, ch = int(W * 0.2), int(H * 0.2)
            x0 = random.randint(0, max(1, W - cw))
            y0 = random.randint(0, max(1, H - ch))
            img_t[:, y0:y0 + ch, x0:x0 + cw] = 0

        # ---- Gaussian blur (small kernels) ----
        if random.random() < 0.3:
            np_img = (img_t.mul(255).permute(1, 2, 0).byte().numpy())
            k = random.choice([3, 5])
            np_img = cv2.GaussianBlur(np_img, (k, k), 0)
            img_t  = torch.from_numpy(np_img).permute(2, 0, 1).float().div(255)

        # ---- Brightness & contrast jitter ----
        b = random.uniform(0.8, 1.2)
        img_t = img_t.mul(b).clamp(0, 1)

        c = random.uniform(0.8, 1.2)
        m = img_t.mean([1, 2], keepdim=True)
        img_t = (img_t - m).mul(c).add(m).clamp(0, 1)

        # ---- Normalize with ImageNet stats ----
        img_t = F.normalize(img_t, mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([img_id])
        }
        return img_t, target


def train_one_epoch(model, loader, optimizer, scheduler, device, λ_loc=2.0):
    """
    Single training epoch: forward → loss dict → weighted sum → step optimizer + scheduler.
    Returns mean loss over the dataloader.
    """
    model.train()
    total = 0.0
    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Faster R-CNN returns a dict of losses in training mode
        loss_dict = model(imgs, tgts)
        loss = loss_dict["loss_classifier"] + λ_loc * loss_dict["loss_box_reg"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, device, λ_loc=2.0):
    """
    Validation pass that keeps model in train() to get comparable loss dicts.
    No gradients are computed. Returns mean validation loss.
    """
    model.train()  # intentional: compute the same loss keys as in training
    total = 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, tgts)
            loss = loss_dict["loss_classifier"] + λ_loc * loss_dict["loss_box_reg"]
            total += loss.item()
    return total / len(loader)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Device selection ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Initialize Weights & Biases ----
    wandb.init(
        project=args.project,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grayscale": True
        }
    )

    # ---- Datasets & loaders ----
    train_ds = CocoFlankDataset(args.train_json, args.images_dir)
    val_ds   = CocoFlankDataset(args.val_json,   args.images_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # ---- Model setup ----
    model = get_model(num_classes=2).to(device)

    # Freeze all backbone params first...
    for p in model.backbone.body.parameters():
        p.requires_grad = False
    # ...then unfreeze the last few stages for fine-tuning
    unfreeze_backbone(model, num_layers=3)

    # ---- Optimizer & LR scheduler ----
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Cosine schedule with 5% warmup (by step, not epoch)
    total_steps = len(train_loader) * args.epochs

    def lr_fn(step):
        warm = total_steps * 0.05
        if step < warm:
            return float(step) / float(max(1, warm))  # linear warmup
        prog = (step - warm) / float(max(1, total_steps - warm))
        return 0.5 * (1.0 + math.cos(math.pi * prog))  # cosine decay

    scheduler = LambdaLR(optimizer, lr_fn)

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        tl = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        vl = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch}/{args.epochs}] train_loss={tl:.4f}  val_loss={vl:.4f}")
        wandb.log({"epoch": epoch, "train_loss": tl, "val_loss": vl})

        # Save a checkpoint every 5 epochs and at the end
        if epoch % 5 == 0 or epoch == args.epochs:
            ck = os.path.join(args.output_dir, f"model_epoch{epoch}.pth")
            torch.save(model.state_dict(), ck)
            print("  • Saved checkpoint:", ck)

    print("✅ Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
