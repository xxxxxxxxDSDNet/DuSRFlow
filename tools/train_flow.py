#!/usr/bin/env python3
"""Train DuFlowNet with real backgrounds and synthetic foreground parallax."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dusrflow.data.flow_dataset import FlowTestDataset, FlowTrainDataset
from dusrflow.flow_utils import PROJECT_ROOT, load_checkpoint, load_config, resolve_path
from dusrflow.metrics import calculate_psnr, calculate_ssim
from dusrflow.models.flow import (
    CharbonnierLoss,
    DuFlowNet,
    EdgeAwareSecondOrderSmoothness,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--foreground-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "flow_training_outputs")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lambda(step, total, minimum=0.01):
    if step >= total:
        return minimum
    return max(minimum, 0.5 * (1.0 + math.cos(math.pi * step / total)))


@torch.no_grad()
def evaluate(model, loader, device, debug=False):
    model.eval()
    psnrs, ssims = [], []
    for index, batch in enumerate(loader):
        lr = batch["lr"].to(device)
        target = batch["target"].to(device)
        reference = batch["reference"].to(device)
        prediction, _ = model(lr, reference)
        prediction_np = prediction[0].clamp(0, 1).mul(255).cpu().numpy().transpose(1, 2, 0)
        target_np = target[0].mul(255).cpu().numpy().transpose(1, 2, 0)
        psnrs.append(calculate_psnr(prediction_np, target_np, crop_border=0))
        ssims.append(calculate_ssim(prediction_np, target_np, crop_border=0))
        if debug and index == 0:
            break
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_root = args.dataset_root or resolve_path(config["dataset_root"])
    dataset_subdir = config.get("dataset_subdir", config["dataset_name"])
    folders = config["folders"]
    required = [
        dataset_root / dataset_subdir / "train" / folders[key]
        for key in ("lr", "target", "reference")
    ]
    required += [
        dataset_root / dataset_subdir / "test" / folders[key]
        for key in ("lr", "target", "reference")
    ]
    required.append(args.foreground_root)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    if args.check_only:
        print(f"Flow training preflight passed: {config['dataset_name']}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DuFlowNet training")

    training = config["training"]
    seed = int(training.get("random_seed", 2024))
    set_seed(seed)
    run_dir = args.output_root / f"{config['dataset_name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(run_dir / "train.log"), logging.StreamHandler()],
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    train_dataset = FlowTrainDataset(
        dataset_root=dataset_root,
        dataset_name=dataset_subdir,
        folders=folders,
        foreground_root=args.foreground_root,
        patch_size=int(training["patch_size"]),
        repeat=int(training.get("repeat", 150)),
        paste_foregrounds=bool(training.get("paste_foregrounds", True)),
        foreground_min=int(training.get("foreground_min", 80)),
        foreground_max=int(training.get("foreground_max", 200)),
        homography=int(training.get("homography", 4)),
        offset=int(training.get("offset", 10)),
        preload=bool(training.get("preload", True)),
        debug=args.debug,
    )
    test_dataset = FlowTestDataset(
        dataset_root,
        dataset_subdir,
        folders,
        args.debug,
        color_order=config.get("test_color_order", "rgb").lower(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1 if args.debug else int(training["batch_size"]),
        shuffle=True,
        num_workers=0 if args.debug else int(training["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    device = torch.device("cuda")
    model = DuFlowNet().to(device)
    if args.resume:
        load_checkpoint(model, args.resume)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    total_iter = 10 if args.debug else int(training["total_iter"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_lambda(step, total_iter)
    )
    reconstruction = CharbonnierLoss()
    smoothness = EdgeAwareSecondOrderSmoothness(alpha=float(training.get("smooth_alpha", 10)))
    smooth_weight = float(training.get("smooth_weight", 0.5))
    test_freq = 10 if args.debug else int(training["test_freq"])
    log_freq = 1 if args.debug else int(training["log_freq"])

    iteration = 0
    model.train()
    while iteration < total_iter:
        for batch in train_loader:
            iteration += 1
            lr = batch["lr"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            reference = batch["reference"].to(device, non_blocking=True)
            prediction, flow = model(lr, reference)
            loss_rec = reconstruction(prediction, target)
            loss_smooth = smoothness(flow, lr)
            loss = loss_rec + smooth_weight * loss_smooth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if iteration % log_freq == 0:
                logging.info(
                    "iter=%d lr=%.7f loss=%.6f rec=%.6f smooth=%.6f",
                    iteration,
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                    loss_rec.item(),
                    loss_smooth.item(),
                )
            if iteration % test_freq == 0 or iteration == total_iter:
                checkpoint = checkpoint_dir / f"net_{iteration}.pth"
                torch.save(model.state_dict(), checkpoint)
                psnr, ssim = evaluate(model, test_loader, device, args.debug)
                logging.info("test iter=%d psnr=%.6f ssim=%.6f checkpoint=%s", iteration, psnr, ssim, checkpoint)
            if iteration >= total_iter:
                break
    print(f"Finished. Output: {run_dir}")


if __name__ == "__main__":
    main()
