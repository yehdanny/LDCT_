"""Training script for LDCT nodule detection.

This module outlines a three-stage training pipeline:

1. Pre-train on the LUNA16 dataset using ``Luna16SliceDataset``.
2. Fine-tune on the hospital 3mm DICOM dataset with ``Hospital3mmDataset``.
3. Evaluate and optionally fine-tune on the 1mm clinical dataset
   ``Clinical1mmDataset``.

Running the full pipeline requires the corresponding datasets to be
placed on disk. Only skeletal training loops are provided; users should
adjust hyperparameters, data augmentations and distributed training
strategies according to their environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou

from .datasets import (
    Luna16SliceDataset,
    Hospital3mmDataset,
    Clinical1mmDataset,
)


def collate_fn(batch):
    images, targets = zip(*batch)
    images = [torch.tensor(img).unsqueeze(0).float() for img in images]
    return images, list(targets)


def train_one_epoch(model, loader: DataLoader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / max(len(loader), 1)


def evaluate_recall(model, loader: DataLoader, device, iou_thresh=0.1) -> float:
    """Compute recall at ``iou_thresh`` over all samples."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                total += len(target["boxes"])
                if len(output["boxes"]) == 0:
                    continue
                ious = box_iou(output["boxes"], target["boxes"])
                max_iou, _ = ious.max(dim=0)
                correct += int((max_iou >= iou_thresh).sum())
    return correct / max(total, 1)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 1: pre-train on LUNA16
    luna_ds = Luna16SliceDataset(args.luna16_root, args.luna16_csv)
    luna_loader = DataLoader(luna_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Pre-training on LUNA16...")
    for epoch in range(args.pretrain_epochs):
        loss = train_one_epoch(model, luna_loader, optimizer, device)
        print(f"LUNA16 epoch {epoch+1}: loss={loss:.4f}")

    # Stage 2: fine-tune on hospital 3mm dataset
    hosp_ds = Hospital3mmDataset(args.hospital_root, args.hospital_csv)
    hosp_loader = DataLoader(hosp_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    print("Fine-tuning on 3mm hospital dataset...")
    for epoch in range(args.ft_epochs):
        loss = train_one_epoch(model, hosp_loader, optimizer, device)
        print(f"Hospital epoch {epoch+1}: loss={loss:.4f}")

    # Stage 3: evaluation on 1mm clinical dataset
    clinical_ds = Clinical1mmDataset(args.clinical_root)
    clinical_loader = DataLoader(clinical_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    recall = evaluate_recall(model, clinical_loader, device)
    print(f"Clinical 1mm recall @0.1 IoU: {recall:.3%}")

    if recall < 0.85:
        print("Recall below target. Consider additional fine-tuning or data augmentation.")
    else:
        print("Recall meets target!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDCT nodule detection")
    parser.add_argument("--luna16-root", required=True, help="Path to LUNA16 data root")
    parser.add_argument("--luna16-csv", required=True, help="CSV annotations for LUNA16")
    parser.add_argument("--hospital-root", required=True, help="Root directory for 3mm DICOMs")
    parser.add_argument("--hospital-csv", required=True, help="Annotation CSV for 3mm dataset")
    parser.add_argument("--clinical-root", required=True, help="Root directory for 1mm patient folders")
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--ft-epochs", type=int, default=1)
    main(parser.parse_args())
