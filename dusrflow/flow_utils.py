"""Shared configuration and checkpoint helpers for DuFlowNet."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return config


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    expanded = os.path.expandvars(str(value))
    if "${" in expanded:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else base / path


def clean_state_dict(checkpoint) -> OrderedDict:
    if isinstance(checkpoint, dict):
        for field in ("state_dict", "model", "net", "params", "params_ema"):
            if isinstance(checkpoint.get(field), dict):
                checkpoint = checkpoint[field]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in checkpoint.items()
    )


def load_checkpoint(model, path: Path):
    state = clean_state_dict(torch.load(path, map_location="cpu"))
    model.load_state_dict(state, strict=True)
