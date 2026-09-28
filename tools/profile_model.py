"""Profile DuSRFlow parameters, FLOPs, peak memory, latency, and FPS.

Usage:
    python -m tools.profile_model --all
    python -m tools.profile_model --params
    python -m tools.profile_model --flops
    python -m tools.profile_model --timing --warmup 10 --repeat 50

The CUDA DCN operator requires an NVIDIA GPU. FLOPs profiling additionally
requires ``thop``. Default LR, LRC, and Ref sizes are 896x448, 448x224, and
896x448, respectively.
"""

import argparse
import importlib
import time
from types import SimpleNamespace

import torch


MODEL_MODULES = {
    "train": "dusrflow.models.archs.DuSRFlow_arch",
    "v2": "dusrflow.models.archs.DuSRFlow_archV2",
    "v3": "dusrflow.models.archs.DuSRFlow_archV3",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Profile DuSRFlow complexity")
    parser.add_argument(
        "--model-variant",
        choices=tuple(MODEL_MODULES),
        default="train",
    )
    parser.add_argument("--flow-checkpoint", default="")
    parser.add_argument("--lr-size", type=int, nargs=2, default=(896, 448))
    parser.add_argument("--center-size", type=int, nargs=2, default=(448, 224))
    parser.add_argument("--ref-size", type=int, nargs=2, default=(896, 448))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--params", action="store_true")
    parser.add_argument("--flops", action="store_true")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def load_model(args, device):
    module = importlib.import_module(MODEL_MODULES[args.model_variant])
    model_args = SimpleNamespace(flownet_weight=args.flow_checkpoint)
    return module.DuSRFlow(model_args).to(device).eval()


def make_inputs(args, device):
    lr_h, lr_w = args.lr_size
    center_h, center_w = args.center_size
    ref_h, ref_w = args.ref_size
    lr = torch.randn(1, 3, lr_h, lr_w, device=device)
    center = torch.randn(1, 3, center_h, center_w, device=device)
    reference = torch.randn(1, 3, ref_h, ref_w, device=device)

    if args.model_variant == "train":
        return lr, center, reference
    if args.model_variant == "v2":
        target = torch.zeros(
            1, 3, max(1, 2 * (lr_h - 16)), max(1, 2 * (lr_w - 16)), device=device
        )
        return lr, center, reference, target
    return lr, center, reference, [2, 2, 2, 2]


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Parameters (total): {total:,}")
    print(f"Parameters (trainable): {trainable:,}")


def calculate_flops(model, inputs):
    try:
        from thop import clever_format, profile
    except ImportError as error:
        raise RuntimeError("FLOPs profiling requires the 'thop' package.") from error

    flops, parameters = profile(model, inputs=inputs, verbose=False)
    formatted_flops, formatted_parameters = clever_format([flops, parameters], "%.3f")
    print(f"FLOPs: {formatted_flops}")
    print(f"THOP parameters: {formatted_parameters}")


def profile_runtime(model, inputs, warmup, repeat):
    for _ in range(warmup):
        with torch.inference_mode():
            model(*inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        with torch.inference_mode():
            model(*inputs)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeat
    print(f"Average inference time: {elapsed_ms:.3f} ms")
    print(f"Throughput: {1000.0 / elapsed_ms:.3f} FPS")


def profile_memory(model, inputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(*inputs)
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated() / (1024**2)
    reserved = torch.cuda.max_memory_reserved() / (1024**2)
    print(f"Peak CUDA memory allocated: {allocated:.2f} MiB")
    print(f"Peak CUDA memory reserved: {reserved:.2f} MiB")


def main():
    args = parse_args()
    if not any((args.params, args.flops, args.memory, args.timing, args.all)):
        args.all = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because DuSRFlow uses CUDA DCN operators.")

    device = torch.device("cuda")
    model = load_model(args, device)
    inputs = make_inputs(args, device)
    print(f"Model variant: {args.model_variant}")
    print(
        f"LR / center / reference: {args.lr_size} / {args.center_size} / {args.ref_size}"
    )

    if args.params or args.all:
        count_parameters(model)
    if args.flops or args.all:
        calculate_flops(model, inputs)
    if args.memory or args.all:
        profile_memory(model, inputs)
    if args.timing or args.all:
        profile_runtime(model, inputs, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
