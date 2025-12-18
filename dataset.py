from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset


def label_from_filename(path: str) -> str:
    name = Path(path).name
    label_re = re.compile(r"_annot\d+_(?P<label>.+?)_image\.(png|jpg|jpeg)$", flags=re.IGNORECASE)

    match = label_re.search(name)
    if not match:
        raise ValueError(f"Cannot parse label from filename: {name}")
    label = match.group("label").strip()
    # Some files have trailing underscores (e.g. 'Ignore_' / 'Region_'); normalize.
    label = label.strip("_")
    return label


def list_image_files(data_folder: str, exts: Sequence[str] = (".png", ".jpg", ".jpeg")) -> list[Path]:
    root = Path(data_folder)
    if not root.exists():
        raise FileNotFoundError(f"DATA_FOLDER not found: {root.resolve()}")

    exts_lc = {e.lower() for e in exts}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_lc]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found under: {root.resolve()}")
    return files


def get_224_crop_boxes(w: int, h: int) -> list[tuple[int, int, int, int]]:
    size = 224*2

    if w < size or h < size:
        return []

    def get_offsets(length: int, crop_size: int) -> list[int]:
        if length == crop_size:
            return [0]
        n = math.ceil(length / crop_size)
        if n == 1:
            return [0]

        step = (length - crop_size) / (n - 1)
        return [round(i * step) for i in range(n)]

    x_offsets = get_offsets(w, size)
    y_offsets = get_offsets(h, size)

    boxes: list[tuple[int, int, int, int]] = []
    for y in y_offsets:
        for x in x_offsets:
            boxes.append((x, y, x + size, y + size))

    return boxes


class FilenameLabelImageDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        transform: Optional[object] = None,
        exclude_classes: Optional[set[str]] = None,
    ):
        self.data_folder = data_folder
        self.transform = transform
        exclude_classes = exclude_classes or set()
        self.exclude_classes = exclude_classes
        exclude_lc = {c.lower() for c in exclude_classes}

        files = list_image_files(data_folder)
        labels = [label_from_filename(p.name) for p in files]
        keep = [i for i, lab in enumerate(labels) if lab.lower() not in exclude_lc]
        self.files = [files[i] for i in keep]
        self.labels_str_files = [labels[i] for i in keep]

        classes = sorted(set(self.labels_str_files))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # Expand each image into 224x224 crops computed from its (w, h).
        samples: list[tuple[Path, tuple[int, int, int, int]]] = []
        labels_str: list[str] = []
        for path, lab in zip(self.files, self.labels_str_files):
            # Reading size only (fast); no full decode here.
            with Image.open(path) as im:
                w, h = im.size
            for box in get_224_crop_boxes(w, h):
                samples.append((path, box))
                labels_str.append(lab)
        self.samples = samples
        self.labels_str = labels_str
        self.targets = [self.class_to_idx[c] for c in self.labels_str]

    def with_transform(self, transform: Optional[object]):
        ds = object.__new__(FilenameLabelImageDataset)
        ds.data_folder = self.data_folder
        ds.transform = transform
        ds.exclude_classes = self.exclude_classes
        ds.files = self.files
        ds.labels_str_files = self.labels_str_files
        ds.class_to_idx = self.class_to_idx
        ds.idx_to_class = self.idx_to_class
        ds.samples = self.samples
        ds.labels_str = self.labels_str
        ds.targets = self.targets
        return ds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, box = self.samples[idx]
        with Image.open(path) as im:
            image = im.convert("RGB")
            image = image.crop(box)
        if self.transform is not None:
            image = self.transform(image)
        y = self.targets[idx]
        return image, y

    def class_counts(self) -> Counter:
        return Counter(self.labels_str)
