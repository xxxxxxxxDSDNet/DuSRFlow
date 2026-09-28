"""Train DuSRFlow with a reconstruction-only or hybrid adversarial loss.

Recommended usage:
    bash scripts/run.sh EXPERIMENT train GPU

Direct example:
    python -m tools.train_sr \
        --dataset-root /path/to/dual_lens_datasets \
        --dataset-name DuSR-RealV2-Paired \
        --dataset-subdir DuSR-RealV2/Paired \
        --dataset-type dusr-realv2 \
        --coordinates ./coordinates/DuSR-RealV2-Paired.json \
        --flow-checkpoint /path/to/DuFlowNet_DuSR-RealV2-Paired.pth \
        --loss l1 \
        --experiment-name DuSR-RealV2-Paired_L1
"""

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from dusrflow.utils import print_args, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train DuSRFlow")
    parser.add_argument("--dataset-root", default="./datasets")
    parser.add_argument("--dataset-name", default="DuSR-RealV2-Paired")
    parser.add_argument(
        "--dataset-subdir",
        default="",
        help="Path below dataset-root; defaults to dataset-name.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=("auto", "dusr-realv2", "external"),
        default="auto",
        help="Dataset layout. 'auto' recognizes the DuSR-RealV2 subsets.",
    )
    parser.add_argument("--coordinates", required=True)
    parser.add_argument(
        "--flow-checkpoint",
        default="",
        help="Pretrained DuFlowNet checkpoint. Empty means random initialization.",
    )
    parser.add_argument("--loss", choices=("l1", "gan"), default="l1")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-root", default="./training_outputs")

    parser.add_argument("--sr-scale", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=2023)
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check-only", action="store_true")

    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument(
        "--nearby-offset-min",
        type=int,
        default=16,
        help="Minimum sampled LR-nearby displacement before applying the multiplier.",
    )
    parser.add_argument(
        "--nearby-offset-max",
        type=int,
        default=96,
        help="Maximum sampled LR-nearby displacement before applying the multiplier.",
    )
    parser.add_argument(
        "--nearby-offset-multiplier",
        type=int,
        default=2,
        help="Multiplier applied to the sampled LR-nearby displacement.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warm-up-iter", type=int, default=10_000)
    parser.add_argument("--total-iter", type=int, default=250_000)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--test-freq", type=int, default=10_000)

    parser.add_argument("--lambda-charbonnier", type=float, default=1.0)
    parser.add_argument("--lambda-perceptual", type=float, default=1e-3)
    parser.add_argument("--lambda-adv", type=float, default=1e-4)

    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-optim", default="")
    parser.add_argument("--resume-scheduler", default="")
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_args(args):
    if args.dataset_type == "auto":
        args.dataset_type = (
            "dusr-realv2"
            if args.dataset_name in {"DuSR-RealV2-Paired", "DuSR-RealV2-Unpaired"}
            else "external"
        )

    # Map public configuration names to the trainer's internal argument names.
    args.flownet_weight = args.flow_checkpoint
    args.use_tb_logger = args.tensorboard
    args.patch_size = list(args.patch_size)
    args.lr = args.learning_rate
    args.loss_Charbonnier = True
    args.loss_perceptual = args.loss == "gan"
    args.loss_adv = args.loss == "gan"
    args.lambda_Charbonnier = args.lambda_charbonnier
    args.lambda_perceptual = args.lambda_perceptual
    args.lambda_adv = args.lambda_adv
    args.exp = args.experiment_name
    args.save_root = args.output_root
    args.testloader = "TestSet"
    return args


def validate_inputs(args):
    if args.nearby_offset_min < 0:
        raise ValueError("--nearby-offset-min must be non-negative.")
    if args.nearby_offset_max < args.nearby_offset_min:
        raise ValueError(
            "--nearby-offset-max must be greater than or equal to "
            "--nearby-offset-min."
        )
    if args.nearby_offset_multiplier <= 0:
        raise ValueError("--nearby-offset-multiplier must be positive.")

    dataset_root = Path(args.dataset_root) / (args.dataset_subdir or args.dataset_name)
    hr_directory = "HR_CC_curve" if args.dataset_type == "dusr-realv2" else "HR"
    train_reference_directory = (
        "HR_SIFT_CC_curve" if args.dataset_type == "dusr-realv2" else "Ref_full"
    )
    reference_directory = "ref" if args.dataset_type == "dusr-realv2" else "Ref_SIFT"
    required = {
        "train/LR": dataset_root / "train" / "LR",
        f"train/{hr_directory}": dataset_root / "train" / hr_directory,
        f"train/{train_reference_directory}": (
            dataset_root / "train" / train_reference_directory
        ),
        "test/LR": dataset_root / "test" / "LR",
        f"test/{hr_directory}": dataset_root / "test" / hr_directory,
        "test/LR_center": dataset_root / "test" / "LR_center",
        f"test/{reference_directory}": (dataset_root / "test" / reference_directory),
        "coordinates": Path(args.coordinates),
    }
    if args.flow_checkpoint:
        required["DuFlowNet checkpoint"] = Path(args.flow_checkpoint)
    if args.resume:
        required["generator checkpoint"] = Path(args.resume)
    if args.resume_optim:
        required["optimizer checkpoint"] = Path(args.resume_optim)
    if args.resume_scheduler:
        required["scheduler checkpoint"] = Path(args.resume_scheduler)

    missing = [
        f"{name}: {path}" for name, path in required.items() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def prepare_output_paths(args):
    output_root = Path(args.output_root)
    if args.resume:
        run_name = Path(args.resume).resolve().parent.parent.name
    else:
        run_name = args.experiment_name

    save_folder = output_root / run_name
    if save_folder.exists() and not args.resume:
        save_folder = output_root / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"

    args.save_folder = str(save_folder)
    args.snapshot_save_dir = str(save_folder / "checkpoints")
    Path(args.snapshot_save_dir).mkdir(parents=True, exist_ok=True)
    return save_folder / "train.log"


def main():
    args = normalize_args(parse_args())
    validate_inputs(args)

    if args.check_only:
        print("Training preflight check passed; no training was run.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DuSRFlow training.")

    log_path = prepare_output_paths(args)
    setup_logger(str(log_path))
    print_args(args)
    set_random_seed(args.random_seed)
    cudnn.benchmark = True

    if not args.flow_checkpoint:
        logging.warning(
            "DuFlowNet is initialized randomly because no checkpoint was given."
        )

    # Import after preflight so --check-only does not compile CUDA extensions.
    from dusrflow.models.sr_trainer import Trainer

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
