#!/usr/bin/env python3
"""Run the train or test stage described by one SR experiment YAML."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run one DuSRFlow YAML experiment")
    parser.add_argument("config", type=Path, help="Path to one experiment YAML")
    parser.add_argument("stage", choices=("train", "test"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--flow-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_environment():
    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("DATASET_ROOT", str(PROJECT_ROOT / "datasets"))
    os.environ.setdefault(
        "FLOW_WEIGHT_ROOT",
        str(PROJECT_ROOT / "weights/pretrained"),
    )
    os.environ.setdefault(
        "TRAIN_OUTPUT_ROOT", str(PROJECT_ROOT / "training_outputs")
    )
    os.environ.setdefault("RESULT_ROOT", str(PROJECT_ROOT / "results"))


def expand(value):
    if isinstance(value, str):
        result = os.path.expandvars(value)
        if "${" in result:
            raise ValueError(f"Unresolved environment variable: {value}")
        return result
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def append_argument(command, key, value):
    option = "--" + key.replace("_", "-")
    if key in {"tensorboard", "save_images", "full_reference_metrics"}:
        command.append(option if value else "--no-" + key.replace("_", "-"))
    elif isinstance(value, bool):
        if value:
            command.append(option)
    elif isinstance(value, list):
        command.append(option)
        command.extend(str(item) for item in value)
    elif value not in (None, ""):
        command.extend((option, str(value)))


def build_command(config, stage, args):
    if stage not in config:
        raise ValueError(f"This experiment has no '{stage}' section.")
    section = config[stage]
    if not isinstance(section, dict):
        raise ValueError(f"'{stage}' must be a mapping.")

    module = section.get("module")
    if not isinstance(module, str) or not module:
        raise ValueError(f"'{stage}.module' must name a Python module.")
    arguments = expand(section.get("arguments", {}))
    if not isinstance(arguments, dict):
        raise ValueError(f"'{stage}.arguments' must be a mapping.")

    if args.dataset_root is not None:
        arguments["dataset_root"] = str(args.dataset_root.expanduser().resolve())
    if args.flow_checkpoint is not None:
        if stage != "train":
            raise ValueError("--flow-checkpoint is only valid for training.")
        arguments["flow_checkpoint"] = str(
            args.flow_checkpoint.expanduser().resolve()
        )
    if args.checkpoint is not None:
        if stage != "test":
            raise ValueError("--checkpoint is only valid for testing.")
        arguments["checkpoint"] = str(args.checkpoint.expanduser().resolve())

    command = [sys.executable, "-m", module]
    for key, value in arguments.items():
        append_argument(command, key, value)
    if args.debug:
        command.append("--debug")
    if args.check_only:
        if stage != "train":
            raise ValueError("--check-only is only supported for training.")
        command.append("--check-only")
    return command


def main():
    args = parse_args()
    configure_environment()

    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Experiment YAML root must be a mapping.")

    command = build_command(config, args.stage, args)
    print(f"Experiment : {config.get('experiment', config_path.stem)}", flush=True)
    print(f"Config     : {config_path}", flush=True)
    print(f"Stage      : {args.stage}", flush=True)
    print(f"GPU        : {args.gpu}", flush=True)
    print(f"Command    : {shlex.join(command)}", flush=True)
    if args.dry_run:
        return

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
