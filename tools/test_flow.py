#!/usr/bin/env python3
"""Run DuFlowNet inference, save warped references, flows, and metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dusrflow.data.flow_dataset import FlowTestDataset
from dusrflow.flow_utils import PROJECT_ROOT, load_checkpoint, load_config, resolve_path
from dusrflow.metrics import calculate_psnr, calculate_ssim
from dusrflow.models.flow import DuFlowNet


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "flow_results")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--save-flow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[:, :, 0] = (angle / 2).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_root = args.dataset_root or resolve_path(config["dataset_root"])
    checkpoint = args.checkpoint or resolve_path(config["checkpoint"])
    dataset_subdir = config.get("dataset_subdir", config["dataset_name"])
    dataset_dir = dataset_root / dataset_subdir / "test"
    required = [checkpoint] + [dataset_dir / config["folders"][key] for key in ("lr", "target", "reference")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    if args.check_only:
        print(f"Flow inference preflight passed: {config['dataset_name']}\nCheckpoint: {checkpoint}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DuFlowNet inference")

    test_mode = config.get("test_mode", "full_lr")
    output_dir = (
        args.output_root / config["dataset_name"] / test_mode / checkpoint.stem
    )
    warped_dir = output_dir / "warped_reference"
    flow_dir = output_dir / "flow"
    flow_vis_dir = output_dir / "flow_visualization"
    warped_dir.mkdir(parents=True, exist_ok=True)
    if args.save_flow:
        flow_dir.mkdir(parents=True, exist_ok=True)
        flow_vis_dir.mkdir(parents=True, exist_ok=True)

    color_order = config.get("test_color_order", "rgb").lower()
    dataset = FlowTestDataset(
        dataset_root,
        dataset_subdir,
        config["folders"],
        args.debug,
        color_order=color_order,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    device = torch.device("cuda")
    model = DuFlowNet().to(device)
    load_checkpoint(model, checkpoint)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=config["dataset_name"]):
            name = batch["name"][0]
            lr = batch["lr"].to(device)
            target = batch["target"].to(device)
            reference = batch["reference"].to(device)
            prediction, flow = model(lr, reference)
            prediction_float = prediction[0].cpu().numpy().transpose(1, 2, 0)
            target_float = target[0].cpu().numpy().transpose(1, 2, 0)
            prediction_uint8 = np.clip(prediction_float * 255.0, 0, 255).astype(
                np.uint8
            )
            prediction_to_save = (
                prediction_uint8
                if color_order == "bgr"
                else cv2.cvtColor(prediction_uint8, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(str(warped_dir / f"{name}.png"), prediction_to_save)
            flow_np = flow[0].cpu().numpy().transpose(1, 2, 0)
            if args.save_flow:
                np.save(flow_dir / f"{name}.npy", flow_np)
                cv2.imwrite(str(flow_vis_dir / f"{name}.png"), flow_to_color(flow_np))
            rows.append(
                {
                    "image_name": name,
                    "psnr": calculate_psnr(
                        prediction_float * 255.0,
                        target_float * 255.0,
                        crop_border=0,
                    ),
                    "ssim": calculate_ssim(
                        prediction_float * 255.0,
                        target_float * 255.0,
                        crop_border=0,
                    ),
                    "flow_mean": float(np.linalg.norm(flow_np, axis=2).mean()),
                    "flow_max": float(np.linalg.norm(flow_np, axis=2).max()),
                }
            )
    summary = {
        "dataset": config["dataset_name"],
        "test_mode": test_mode,
        "color_order": color_order,
        "metric_domain": "float",
        "checkpoint": str(checkpoint.resolve()),
        "num_images": len(rows),
        "psnr": float(np.mean([row["psnr"] for row in rows])),
        "ssim": float(np.mean([row["ssim"] for row in rows])),
        "flow_mean": float(np.mean([row["flow_mean"] for row in rows])),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with (output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
