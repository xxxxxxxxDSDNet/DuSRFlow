"""Evaluate DuSRFlow and report PNG-domain PSNR, SSIM, and LPIPS.

Recommended usage:
    bash scripts/run.sh EXPERIMENT test GPU

Direct example:
    python -m tools.test_sr \
        --dataset-root /path/to/dual_lens_datasets \
        --dataset-name DuSR-RealV2-Paired \
        --dataset-subdir DuSR-RealV2/Paired \
        --dataset-type dusr-realv2 \
        --checkpoint ./weights/pretrained/DuSRFlow_DuSR-RealV2-Paired_L1.pth \
        --coordinates ./coordinates/DuSR-RealV2-Paired.json \
        --result-dir ./results \
        --run-name Paired_L1

Outputs:
    <result-dir>/<dataset-name>/<run-name>/<checkpoint-name>/
    ├── *.png
    ├── metrics.json
    └── per_image_metrics.csv

Timestamped logs and output.txt are stored one directory above the images.
"""

import argparse
import csv
import importlib
import json
import logging
import time
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from dusrflow.data.sr_dataset import TestSet
from dusrflow.checkpoints import migrate_legacy_sr_state_dict
from dusrflow.metrics import calculate_psnr, calculate_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DuSRFlow")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--dataset-subdir",
        default="",
        help="Path below dataset-root; defaults to dataset-name.",
    )
    parser.add_argument(
        "--dataset-type",
        required=True,
        choices=("dusr-realv2", "external"),
        help="Use 'dusr-realv2' for DuSR-RealV2 and 'external' otherwise.",
    )
    parser.add_argument(
        "--model-variant",
        choices=("auto", "v0", "v2", "v3"),
        default="auto",
        help=(
            "Select the inference implementation. V2 supports both "
            "even and native patch-grid phases."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--sr-scale", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--patch-phase",
        choices=("even", "native"),
        default="even",
        help=(
            "Patch-grid phase used by the V2 evaluator. 'even' reproduces "
            "the later parity-adjusted evaluator; 'native' preserves the "
            "original patch origins used by the V3 evaluator."
        ),
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Require the checkpoint keys to match the selected model exactly.",
    )
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save reconstructed PNG images.",
    )
    parser.add_argument(
        "--metric-domain",
        choices=("png", "float"),
        default="png",
        help=("PSNR/SSIM domain. LPIPS always uses quantized PNG-domain inputs."),
    )
    parser.add_argument(
        "--full-reference-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute PSNR, SSIM, and LPIPS. Disable for Unpaired inference.",
    )
    parser.add_argument("--debug", action="store_true", help="Evaluate one sample.")
    return parser.parse_args()


def configure_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )


def validate_inputs(args):
    dataset_dir = (
        Path(args.dataset_root) / (args.dataset_subdir or args.dataset_name) / "test"
    )
    required_paths = {
        "test dataset": dataset_dir,
        "checkpoint": Path(args.checkpoint),
        "coordinates": Path(args.coordinates),
    }
    missing = [
        f"{name}: {path}" for name, path in required_paths.items() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def load_test_dataset(args):
    return TestSet(args)


def load_model_class(args):
    variant = args.model_variant
    if variant == "auto":
        variant = "v2" if args.dataset_type == "dusr-realv2" else "v3"
    if variant == "v0":
        # Compute the global correlation matrix one query row at a time.
        args.inference_matching_row = 1
    suffix = "" if variant == "v0" else variant.upper()
    module_name = f"dusrflow.models.archs.DuSRFlow_arch{suffix}"
    module = importlib.import_module(module_name)
    return module.DuSRFlow, variant


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    for key in ("state_dict", "model", "net", "params", "params_ema"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            logging.info("Reading network parameters from checkpoint field '%s'.", key)
            break

    return migrate_legacy_sr_state_dict(checkpoint)


def load_checkpoint(model, checkpoint_path, strict):
    logging.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        if incompatible.missing_keys:
            logging.warning(
                "Checkpoint is missing %d model keys: %s",
                len(incompatible.missing_keys),
                incompatible.missing_keys[:10],
            )
        if incompatible.unexpected_keys:
            logging.warning(
                "Checkpoint contains %d unused keys: %s",
                len(incompatible.unexpected_keys),
                incompatible.unexpected_keys[:10],
            )
        if not incompatible.missing_keys and not incompatible.unexpected_keys:
            logging.info("Checkpoint keys match the selected model.")


def prepare_batch(batch, device):
    required_keys = ("LR", "LR_center", "Ref_SIFT", "HR", "name")
    missing = [key for key in required_keys if key not in batch]
    if missing:
        raise KeyError(f"Batch is missing required fields: {missing}")

    return {
        key: value.to(device, non_blocking=True) if key != "name" else value
        for key, value in batch.items()
    }


def adjust_reference_size(lr_center, reference, coordinate):
    if (coordinate[0] // 2) % 2 == 1:
        lr_center = lr_center[:, :, 1:-1, :]
        reference = reference[:, :, 2:-2, :]
    if (coordinate[2] // 2) % 2 == 1:
        lr_center = lr_center[:, :, :, 1:-1]
        reference = reference[:, :, :, 2:-2]
    return lr_center, reference


def run_inference(model, variant, lr, lr_center, reference, target, coordinate):
    """Run inference with the selected architecture variant."""
    if variant == "v0":
        # The original full-image evaluator pads LR so the internal patch grid
        # is exactly divisible by (LRC height/width - 16), then removes only
        # that extra padding from the x2 output. This path is required by the
        # This path is required by the DuSR-Real hybrid-loss checkpoint.
        patch_height = lr_center.shape[-2] - 16
        patch_width = lr_center.shape[-1] - 16
        if patch_height <= 0 or patch_width <= 0:
            raise ValueError("LR-center must be larger than 16 pixels.")
        extra_height = (-lr.shape[-2]) % patch_height
        extra_width = (-lr.shape[-1]) % patch_width
        lr = F.pad(lr, (0, extra_width, 0, extra_height), mode="reflect")
        output = model(nn.ReplicationPad2d(8)(lr), lr_center, reference)
        if extra_height:
            output = output[:, :, : -2 * extra_height, :]
        if extra_width:
            output = output[:, :, :, : -2 * extra_width]
        return output

    lr = nn.ReplicationPad2d(8)(lr)
    context = target if variant == "v2" else coordinate
    return model(lr, lr_center, reference, context)


def average(values):
    return float(np.mean(values)) if values else float("nan")


def quantize_like_png(image):
    """Match torchvision.save_image's clamp, round, and uint8 conversion."""
    return image.clamp(0.0, 1.0).mul(255.0).add(0.5).clamp(0.0, 255.0).to(torch.uint8)


def prepare_metric_images(output, target, metric_domain):
    """Convert tensors to the image domain used by PSNR and SSIM."""
    if metric_domain == "png":
        output_image = quantize_like_png(output)[0]
        target_image = quantize_like_png(target)[0]
    elif metric_domain == "float":
        output_image = output[0].mul(255.0)
        target_image = target[0].mul(255.0)
    else:
        raise ValueError(f"Unsupported metric domain: {metric_domain}")

    return (
        output_image.cpu().numpy().transpose(1, 2, 0),
        target_image.cpu().numpy().transpose(1, 2, 0),
    )


def prepare_png_lpips_tensors(output, target, device):
    """Match AFtip/Metrics/metric.py's LPIPS input pipeline."""
    output_image, target_image = prepare_metric_images(output, target, "png")
    output_tensor = lpips.im2tensor(output_image).to(device)
    target_tensor = lpips.im2tensor(target_image).to(device)
    return output_tensor, target_tensor


def save_summaries(
    run_dir,
    checkpoint_path,
    dataset_name,
    model_variant,
    patch_phase,
    metric_domain,
    rows,
):
    summary = {
        "dataset": dataset_name,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "model_variant": model_variant,
        "patch_phase": patch_phase if model_variant == "v2" else None,
        "num_images": len(rows),
        "psnr_ssim_metric_domain": metric_domain,
        "lpips_metric_domain": "png",
        "lpips_backbone": "alex",
        "all": {
            "psnr": average([row["psnr_all"] for row in rows]),
            "ssim": average([row["ssim_all"] for row in rows]),
            "lpips": average([row["lpips"] for row in rows]),
        },
        "center": {
            "psnr": average([row["psnr_center"] for row in rows]),
            "ssim": average([row["ssim_center"] for row in rows]),
        },
        "corner": {
            "psnr": average([row["psnr_corner"] for row in rows]),
            "ssim": average([row["ssim_corner"] for row in rows]),
        },
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    with (run_dir / "per_image_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output_path = run_dir.parent / "output.txt"
    with output_path.open("a", encoding="utf-8") as file:
        file.write(f"\n{run_dir.name}\n")
        file.write(
            "ALL:     psnr:{psnr:.6f}   ssim:{ssim:.6f}  lpips:{lpips:.6f}\n".format(
                **summary["all"]
            )
        )
        file.write(
            "CENTER:  psnr:{psnr:.6f}   ssim:{ssim:.6f}\n".format(**summary["center"])
        )
        file.write(
            "CORNER:  psnr:{psnr:.6f}   ssim:{ssim:.6f}\n".format(**summary["corner"])
        )
    return summary


def main():
    args = parse_args()
    validate_inputs(args)

    checkpoint_name = Path(args.checkpoint).stem
    run_dir = (
        Path(args.result_dir) / args.dataset_name / args.run_name / checkpoint_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir.parent / f"{time.strftime('%Y%m%d_%H%M%S')}.log"
    configure_logging(log_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DuSRFlow evaluation.")

    device = torch.device("cuda")
    cudnn.benchmark = True

    logging.info("Dataset: %s", args.dataset_name)
    logging.info("Dataset type: %s", args.dataset_type)
    logging.info("Result directory: %s", run_dir)
    logging.info("PSNR/SSIM metric domain: %s", args.metric_domain)

    test_dataset = load_test_dataset(args)
    if len(test_dataset) == 0:
        raise RuntimeError("The test dataset is empty.")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    logging.info("Number of test images: %d", len(test_dataset))

    with Path(args.coordinates).open("r", encoding="utf-8") as file:
        coordinates = json.load(file)

    model_class, model_variant = load_model_class(args)
    logging.info("Model variant: %s", model_variant)
    model = model_class(args)
    load_checkpoint(model, args.checkpoint, args.strict_load)
    model = model.to(device).eval()
    lpips_model = None
    if args.full_reference_metrics:
        lpips_model = lpips.LPIPS(net="alex", spatial=True).to(device).eval()

    rows = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc=args.dataset_name):
            batch = prepare_batch(batch, device)
            image_name = batch["name"][0]
            if image_name not in coordinates:
                raise KeyError(f"Coordinates do not contain image: {image_name}")

            coordinate = coordinates[image_name]
            lr_center, reference = adjust_reference_size(
                batch["LR_center"], batch["Ref_SIFT"], coordinate
            )
            output = run_inference(
                model,
                model_variant,
                batch["LR"],
                lr_center,
                reference,
                batch["HR"],
                coordinate,
            )
            if isinstance(output, dict):
                output = output["output"]
            output = output.clamp(0.0, 1.0)

            if args.save_images:
                torchvision.utils.save_image(output[0], run_dir / image_name)

            if args.full_reference_metrics:
                lpips_output, lpips_target = prepare_png_lpips_tensors(
                    output, batch["HR"], device
                )
                lpips_value = lpips_model(lpips_output, lpips_target).mean().item()
                output_image, target_image = prepare_metric_images(
                    output, batch["HR"], args.metric_domain
                )
                psnr = calculate_psnr(output_image, target_image, coordinate)
                ssim = calculate_ssim(output_image, target_image, coordinate)

                row = {
                    "image": image_name,
                    "psnr_all": float(psnr[0]),
                    "ssim_all": float(ssim[0]),
                    "lpips": float(lpips_value),
                    "psnr_center": float(psnr[1]),
                    "ssim_center": float(ssim[1]),
                    "psnr_corner": float(psnr[2]),
                    "ssim_corner": float(ssim[2]),
                }
                rows.append(row)
                logging.info(
                    (
                        "%s | ALL PSNR %.6f SSIM %.6f LPIPS %.6f"
                        " | CENTER PSNR %.6f SSIM %.6f"
                        " | CORNER PSNR %.6f SSIM %.6f"
                    ),
                    image_name,
                    row["psnr_all"],
                    row["ssim_all"],
                    row["lpips"],
                    row["psnr_center"],
                    row["ssim_center"],
                    row["psnr_corner"],
                    row["ssim_corner"],
                )
            else:
                rows.append({"image": image_name})
                logging.info("%s | inference complete", image_name)

    if not args.full_reference_metrics:
        manifest = {
            "dataset": args.dataset_name,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "model_variant": model_variant,
            "num_images": len(rows),
            "full_reference_metrics": False,
            "images": [row["image"] for row in rows],
        }
        with (run_dir / "inference_manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
        logging.info("Inference complete: %d images", len(rows))
        return

    summary = save_summaries(
        run_dir,
        args.checkpoint,
        args.dataset_name,
        model_variant,
        args.patch_phase,
        args.metric_domain,
        rows,
    )
    logging.info(
        "Dataset average | ALL PSNR %.6f SSIM %.6f LPIPS %.6f",
        summary["all"]["psnr"],
        summary["all"]["ssim"],
        summary["all"]["lpips"],
    )
    logging.info(
        "Dataset average | CENTER PSNR %.6f SSIM %.6f",
        summary["center"]["psnr"],
        summary["center"]["ssim"],
    )
    logging.info(
        "Dataset average | CORNER PSNR %.6f SSIM %.6f",
        summary["corner"]["psnr"],
        summary["corner"]["ssim"],
    )


if __name__ == "__main__":
    main()
