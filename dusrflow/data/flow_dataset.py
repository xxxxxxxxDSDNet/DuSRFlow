"""Datasets for real-background/synthetic-foreground DuFlowNet training."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def index_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def matched_triplets(root: Path, split: str, folders: dict[str, str]):
    indexed = {
        key: index_images(root / split / folder) for key, folder in folders.items()
    }
    common = set.intersection(*(set(images) for images in indexed.values()))
    if not common:
        raise RuntimeError(f"No matched image triplets under {root / split}")
    for key, images in indexed.items():
        missing = len(
            set.union(*(set(item) for item in indexed.values())) - set(images)
        )
        if missing:
            print(f"Warning: {key} is missing {missing} filenames in {root / split}")
    return [
        (name, *(indexed[key][name] for key in ("lr", "target", "reference")))
        for name in sorted(common)
    ]


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_bgr(path: Path) -> np.ndarray:
    """Read an image without converting OpenCV's BGR channel order."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def to_tensor(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(image.astype(np.float32) / 255.0)
    return torch.from_numpy(image).permute(2, 0, 1)


def resize_like(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    if image.shape[:2] == target.shape[:2]:
        return image
    return cv2.resize(
        image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR
    )


class FlowTrainDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        dataset_name: str,
        folders: dict[str, str],
        foreground_root: Path,
        patch_size: int,
        repeat: int,
        paste_foregrounds: bool,
        foreground_min: int,
        foreground_max: int,
        homography: int,
        offset: int,
        preload: bool = True,
        debug: bool = False,
    ):
        self.samples = matched_triplets(dataset_root / dataset_name, "train", folders)
        if debug:
            self.samples = self.samples[:4]
        self.patch_size = patch_size
        self.repeat = 1 if debug else repeat
        self.paste_foregrounds = paste_foregrounds
        self.foreground_min = foreground_min
        self.foreground_max = foreground_max
        self.homography = homography
        self.offset = offset
        self.preload = preload
        self.cached_samples = None
        if self.preload:
            self.cached_samples = []
            for name, lr_path, target_path, reference_path in tqdm(
                self.samples, desc="Preload Flow train images"
            ):
                lr = read_rgb(lr_path)
                target = resize_like(read_rgb(target_path), lr)
                reference = resize_like(read_rgb(reference_path), lr)
                self.cached_samples.append((name, lr, target, reference))
        self.foregrounds = []
        if paste_foregrounds:
            paths = sorted(
                path for path in foreground_root.glob("*.png") if path.is_file()
            )
            if not paths:
                raise FileNotFoundError(
                    f"No RGBA foreground PNGs found in {foreground_root}. "
                    "Set FLOW_FOREGROUND_ROOT or pass --foreground-root."
                )
            for path in paths:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None or image.ndim != 3 or image.shape[2] != 4:
                    raise ValueError(f"Foreground must be an RGBA PNG: {path}")
                self.foregrounds.append(image)

    def __len__(self):
        return len(self.samples) * self.repeat

    def _crop(self, lr, target, reference):
        height, width = lr.shape[:2]
        if min(height, width) < self.patch_size:
            raise ValueError(
                f"Image {width}x{height} is smaller than patch {self.patch_size}"
            )
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        sl = np.s_[top : top + self.patch_size, left : left + self.patch_size]
        return lr[sl].copy(), target[sl].copy(), reference[sl].copy()

    @staticmethod
    def _augment(*images):
        if random.random() < 0.5:
            images = tuple(image[:, ::-1] for image in images)
        if random.random() < 0.5:
            images = tuple(image[::-1] for image in images)
        rotations = random.randint(0, 2)
        return tuple(np.rot90(image, rotations).copy() for image in images)

    def _perspective_pair(self, rgba: np.ndarray, height: int, width: int):
        src = np.float32(
            [
                [self.homography, self.homography],
                [width - 1 - self.homography, self.homography],
                [width - 1 - self.homography, height - 1 - self.homography],
                [self.homography, height - 1 - self.homography],
            ]
        )

        def transform():
            jitter = np.random.randint(-self.homography, self.homography, size=(4, 2))
            dst = src + jitter.astype(np.float32)
            matrix = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(rgba, matrix, (width, height))

        return transform(), transform()

    @staticmethod
    def _composite(background, foreground, top, left):
        height, width = foreground.shape[:2]
        rgb = cv2.cvtColor(foreground[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = foreground[:, :, 3:4].astype(np.float32) / 255.0
        region = background[top : top + height, left : left + width]
        background[top : top + height, left : left + width] = (
            region * (1.0 - alpha) + rgb * alpha
        )

    def _paste(self, lr, target, reference):
        for _ in range(2):
            height = random.randint(self.foreground_min, self.foreground_max - 1)
            width = random.randint(self.foreground_min, self.foreground_max - 1)
            rgba = random.choice(self.foregrounds)
            max_top = rgba.shape[0] - height
            max_left = rgba.shape[1] - width
            if max_top < 0 or max_left < 0:
                continue
            top_src = random.randint(0, max_top)
            left_src = random.randint(0, max_left)
            crop = rgba[top_src : top_src + height, left_src : left_src + width]
            center_fg, reference_fg = self._perspective_pair(crop, height, width)
            center_rgb = cv2.cvtColor(center_fg[:, :, :3], cv2.COLOR_BGR2RGB)
            degraded = cv2.resize(
                center_rgb,
                (width // 2, height // 2),
                interpolation=random.choice(
                    (cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LINEAR)
                ),
            )
            degraded = cv2.resize(
                degraded,
                (width, height),
                interpolation=random.choice(
                    (cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_LINEAR)
                ),
            )
            lr_foreground = center_fg.copy()
            lr_foreground[:, :, :3] = cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR)
            top = random.randint(self.offset, self.patch_size - height - self.offset)
            left = random.randint(self.offset, self.patch_size - width - self.offset)
            shift_y = random.randint(-self.offset, self.offset - 1)
            shift_x = random.randint(-self.offset, self.offset - 1)
            self._composite(lr, lr_foreground, top, left)
            self._composite(target, center_fg, top, left)
            self._composite(reference, reference_fg, top + shift_y, left + shift_x)
        return lr, target, reference

    def __getitem__(self, index):
        sample_index = index % len(self.samples)
        if self.cached_samples is not None:
            _, lr, target, reference = self.cached_samples[sample_index]
        else:
            _, lr_path, target_path, reference_path = self.samples[sample_index]
            lr = read_rgb(lr_path)
            target = resize_like(read_rgb(target_path), lr)
            reference = resize_like(read_rgb(reference_path), lr)
        lr, target, reference = self._crop(lr, target, reference)
        lr, target, reference = self._augment(lr, target, reference)
        if self.paste_foregrounds:
            lr, target, reference = self._paste(lr, target, reference)
        return {
            "lr": to_tensor(lr),
            "target": to_tensor(target),
            "reference": to_tensor(reference),
        }


class FlowTestDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        dataset_name: str,
        folders: dict[str, str],
        debug: bool = False,
        color_order: str = "rgb",
        preload: bool = True,
    ):
        self.samples = matched_triplets(dataset_root / dataset_name, "test", folders)
        if color_order not in {"rgb", "bgr"}:
            raise ValueError(f"Unsupported Flow test color order: {color_order}")
        self.read_image = read_bgr if color_order == "bgr" else read_rgb
        if debug:
            self.samples = self.samples[:4]
        self.cached_samples = None
        if preload:
            self.cached_samples = []
            for name, lr_path, target_path, reference_path in tqdm(
                self.samples, desc="Preload Flow test images"
            ):
                lr = self.read_image(lr_path)
                target = resize_like(self.read_image(target_path), lr)
                reference = resize_like(self.read_image(reference_path), lr)
                self.cached_samples.append((name, lr, target, reference))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.cached_samples is not None:
            name, lr, target, reference = self.cached_samples[index]
        else:
            name, lr_path, target_path, reference_path = self.samples[index]
            lr = self.read_image(lr_path)
            target = resize_like(self.read_image(target_path), lr)
            reference = resize_like(self.read_image(reference_path), lr)
        return {
            "name": name,
            "lr": to_tensor(lr),
            "target": to_tensor(target),
            "reference": to_tensor(reference),
        }
