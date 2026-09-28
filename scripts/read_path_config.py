#!/usr/bin/env python3
"""Read selected values from the flat local-path YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_flat_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid entry at {path}:{line_number}: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid entry at {path}:{line_number}: {raw_line}")
        values[key] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("keys", nargs="+")
    args = parser.parse_args()

    values = read_flat_yaml(args.config)
    missing = [key for key in args.keys if key not in values]
    if missing:
        raise KeyError(f"Missing path configuration: {', '.join(missing)}")
    for key in args.keys:
        print(values[key])


if __name__ == "__main__":
    main()
