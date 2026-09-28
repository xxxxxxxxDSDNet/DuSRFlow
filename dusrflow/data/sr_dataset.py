"""DuSRFlow training and evaluation datasets."""

from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


DATASET_LAYOUTS = {
    "dusr-realv2": {
        "hr": "HR_CC_curve",
        "train_reference": "HR_SIFT_CC_curve",
        "test_reference": "ref",
    },
    "external": {
        "hr": "HR",
        "train_reference": "Ref_full",
        "test_reference": "Ref_SIFT",
    },
}


def resolve_dataset_type(args):
    dataset_type = getattr(args, "dataset_type", "auto")
    if dataset_type == "auto":
        return (
            "dusr-realv2"
            if args.dataset_name in {"DuSR-RealV2-Paired", "DuSR-RealV2-Unpaired"}
            else "external"
        )
    if dataset_type not in DATASET_LAYOUTS:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return dataset_type


def list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    images = sorted(path for path in directory.iterdir() if path.is_file())
    if not images:
        raise RuntimeError(f"No images found in: {directory}")
    return images


def aligned_image_lists(named_directories):
    lists = {name: list_images(path) for name, path in named_directories.items()}
    reference_name = next(iter(lists))
    reference_files = [path.name for path in lists[reference_name]]

    for name, paths in lists.items():
        filenames = [path.name for path in paths]
        if filenames != reference_files:
            missing = sorted(set(reference_files) - set(filenames))
            extra = sorted(set(filenames) - set(reference_files))
            raise RuntimeError(
                f"Dataset files are not aligned for {name}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    return lists


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a three-channel image: {path}")
    return image[..., ::-1]


def image_to_tensor(image):
    array = np.ascontiguousarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).float()


class TrainSet(Dataset):
    """Cache aligned training images and sample paired random patches."""

    def __init__(self, args):
        self.args = args
        self.scale = args.sr_scale
        dataset_type = resolve_dataset_type(args)
        layout = DATASET_LAYOUTS[dataset_type]
        dataset_subdir = getattr(args, "dataset_subdir", "") or args.dataset_name
        train_root = Path(args.dataset_root) / dataset_subdir / "train"

        paths = aligned_image_lists(
            {
                "LR": train_root / "LR",
                "HR": train_root / layout["hr"],
                "Reference": train_root / layout["train_reference"],
            }
        )
        if args.debug:
            paths = {name: values[:8] for name, values in paths.items()}

        self.lr_images = []
        self.hr_images = []
        self.reference_images = []
        for lr_path, hr_path, reference_path in tqdm(
            zip(paths["LR"], paths["HR"], paths["Reference"]),
            total=len(paths["LR"]),
            desc=f"Cache train/{args.dataset_name}",
        ):
            self.lr_images.append(read_rgb(lr_path))
            self.hr_images.append(read_rgb(hr_path))
            self.reference_images.append(read_rgb(reference_path))

    def __len__(self):
        return len(self.lr_images) * 50

    def crop_patch(self, lr, hr, reference, patch_size):
        height, width = lr.shape[:2]
        patch_h, patch_w = patch_size
        if height < patch_h or width < patch_w:
            raise ValueError(
                f"Patch {patch_size} exceeds LR image size {(height, width)}"
            )

        left = random.randint(0, width - patch_w)
        top = random.randint(0, height - patch_h)
        hr_left, hr_top = self.scale * left, self.scale * top
        hr_patch_h, hr_patch_w = self.scale * patch_h, self.scale * patch_w

        # The experiment YAML explicitly controls the LR-nearby displacement.
        offset_h = (
            random.randint(self.args.nearby_offset_min, self.args.nearby_offset_max)
            * self.args.nearby_offset_multiplier
        )
        offset_w = (
            random.randint(self.args.nearby_offset_min, self.args.nearby_offset_max)
            * self.args.nearby_offset_multiplier
        )
        nearby_top = top + offset_h * random.choice((-1, 1))
        nearby_left = left + offset_w * random.choice((-1, 1))
        nearby_top = max(10, min(nearby_top, height - patch_h - 10))
        nearby_left = max(10, min(nearby_left, width - patch_w - 10))
        ref_top = self.scale * nearby_top
        ref_left = self.scale * nearby_left

        return (
            lr[top : top + patch_h, left : left + patch_w],
            hr[hr_top : hr_top + hr_patch_h, hr_left : hr_left + hr_patch_w],
            lr[
                nearby_top : nearby_top + patch_h,
                nearby_left : nearby_left + patch_w,
            ],
            reference[
                ref_top : ref_top + hr_patch_h,
                ref_left : ref_left + hr_patch_w,
            ],
        )

    @staticmethod
    def augment(*images):
        horizontal_flip = random.random() < 0.5
        vertical_flip = random.random() < 0.5
        rotation = np.random.randint(0, 3)

        def apply(image):
            if horizontal_flip:
                image = image[:, ::-1, :]
            if vertical_flip:
                image = image[::-1, :, :]
            return np.rot90(image, rotation)

        return [apply(image) for image in images]

    def __getitem__(self, index):
        index %= len(self.lr_images)
        images = self.crop_patch(
            self.lr_images[index],
            self.hr_images[index],
            self.reference_images[index],
            self.args.patch_size,
        )
        lr, hr, lr_nearby, reference = self.augment(*images)
        return {
            "lr": image_to_tensor(lr),
            "lr_nearby": image_to_tensor(lr_nearby),
            "ref": image_to_tensor(reference),
            "hr": image_to_tensor(hr),
        }


class TestSet(Dataset):
    """Cache the complete test split for patch-based evaluation."""

    def __init__(self, args):
        dataset_type = resolve_dataset_type(args)
        layout = DATASET_LAYOUTS[dataset_type]
        dataset_subdir = getattr(args, "dataset_subdir", "") or args.dataset_name
        test_root = Path(args.dataset_root) / dataset_subdir / "test"

        paths = aligned_image_lists(
            {
                "LR": test_root / "LR",
                "HR": test_root / layout["hr"],
                "LR_center": test_root / "LR_center",
                "Ref_SIFT": test_root / layout["test_reference"],
            }
        )
        if args.debug:
            paths = {name: values[:1] for name, values in paths.items()}

        self.names = [path.name for path in paths["LR"]]
        self.lr_images = []
        self.hr_images = []
        self.lr_center_images = []
        self.reference_images = []
        for lr_path, hr_path, center_path, reference_path in tqdm(
            zip(
                paths["LR"],
                paths["HR"],
                paths["LR_center"],
                paths["Ref_SIFT"],
            ),
            total=len(paths["LR"]),
            desc=f"Cache test/{args.dataset_name}",
        ):
            self.lr_images.append(read_rgb(lr_path))
            self.hr_images.append(read_rgb(hr_path))
            self.lr_center_images.append(read_rgb(center_path))
            self.reference_images.append(read_rgb(reference_path))

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, index):
        return {
            "LR": image_to_tensor(self.lr_images[index]),
            "HR": image_to_tensor(self.hr_images[index]),
            "LR_center": image_to_tensor(self.lr_center_images[index]),
            "Ref_SIFT": image_to_tensor(self.reference_images[index]),
            "name": self.names[index],
        }


# Backward-compatible dataset alias.
TestSet_cache = TestSet
