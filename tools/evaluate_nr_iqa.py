#!/usr/bin/env python3
"""Evaluate no-reference image-quality metrics on an SR result folder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyiqa
import torch
from tqdm import tqdm


DEFAULT_METRICS = (
    "musiq",
    "maniqa",
    "clipiqa",
    "liqe",
    "nrqm",
    "niqe",
    "pi",
    "brisque",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    images = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No images found in: {args.input_dir}")

    device = torch.device(args.device)
    evaluators = {
        name: pyiqa.create_metric(name, device=device) for name in args.metrics
    }
    rows = []
    with torch.inference_mode():
        for image_path in tqdm(images, desc="No-reference IQA"):
            row = {"image": image_path.name}
            for name, evaluator in evaluators.items():
                row[name] = float(evaluator(str(image_path)).item())
            rows.append(row)

    averages = {
        name: float(np.mean([row[name] for row in rows])) for name in args.metrics
    }
    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "num_images": len(rows),
        "metrics": list(args.metrics),
        "average": averages,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "nr_iqa_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "nr_iqa_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
