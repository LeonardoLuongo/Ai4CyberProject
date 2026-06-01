"""Shared utilities for adversarial image generation.

The scripts in this project use PyTorch and ART with the course NN1 model:

    facenet_pytorch.InceptionResnetV1(pretrained="vggface2", classify=True)

The generated adversarial images are saved as PNG files. Inputs are resized to
160x160 only inside this attack pipeline because NN1 requires that size.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_ALLOWED_EPS = 0.201


@dataclass(frozen=True)
class AttackSample:
    image_path: Path
    relative_path: Path
    identity_dir: str
    dataset_label: int | None
    identity_id: str | None
    identity_name: str | None


def validate_eps(eps: float) -> None:
    if eps <= 0:
        raise ValueError("--eps must be greater than 0")
    if eps > MAX_ALLOWED_EPS:
        raise ValueError(
            f"--eps={eps} exceeds the project limit {MAX_ALLOWED_EPS}. "
            "NN1 inputs are in [0, 1], so 10% of the representable range is 0.1."
        )


def parse_identity_dir(identity_dir: str) -> tuple[int | None, str | None, str | None]:
    """Parse folders such as 000_n007126_n007126."""
    match = re.match(r"^(?P<label>\d+?)_(?P<identity_id>n\d+)_(?P<name>.+)$", identity_dir)
    if not match:
        return None, None, None

    return (
        int(match.group("label")),
        match.group("identity_id"),
        match.group("name"),
    )


def discover_test_images(input_dir: Path) -> list[AttackSample]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")

    samples: list[AttackSample] = []
    for image_path in sorted(input_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = image_path.relative_to(input_dir)
        identity_dir = relative_path.parts[0] if len(relative_path.parts) > 1 else image_path.parent.name
        dataset_label, identity_id, identity_name = parse_identity_dir(identity_dir)
        samples.append(
            AttackSample(
                image_path=image_path,
                relative_path=relative_path,
                identity_dir=identity_dir,
                dataset_label=dataset_label,
                identity_id=identity_id,
                identity_name=identity_name,
            )
        )

    if not samples:
        raise ValueError(f"no images found under {input_dir}")

    return samples


def batched(items: list[AttackSample], batch_size: int) -> Iterable[list[AttackSample]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_image_for_nn1(image_path: Path, image_size: int) -> np.ndarray:
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"cannot read image {image_path}") from exc

    return np.transpose(array, (2, 0, 1))


def load_batch_for_nn1(samples: list[AttackSample], image_size: int) -> np.ndarray:
    arrays = [load_image_for_nn1(sample.image_path, image_size) for sample in samples]
    return np.stack(arrays, axis=0).astype(np.float32)


def save_adv_image(chw_array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(chw_array, 0.0, 1.0)
    hwc = np.transpose(clipped, (1, 2, 0))
    image = Image.fromarray(np.rint(hwc * 255.0).astype(np.uint8), mode="RGB")
    image.save(output_path, format="PNG")


def output_path_for_sample(output_dir: Path, sample: AttackSample) -> Path:
    relative_parent = sample.relative_path.parent
    filename = f"{sample.image_path.stem}_adv.png"
    return output_dir / relative_parent / filename


def clean_reference_path_for_sample(output_dir: Path, sample: AttackSample) -> Path:
    relative_parent = sample.relative_path.parent
    filename = f"{sample.image_path.stem}_clean_nn1.png"
    return output_dir / "clean_reference_nn1" / relative_parent / filename


def perturbation_stats(original: np.ndarray, adversarial: np.ndarray) -> tuple[float, float]:
    perturbation = np.abs(adversarial - original)
    return float(np.max(perturbation)), float(np.mean(perturbation))


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((labels.size, num_classes), dtype=np.float32)
    encoded[np.arange(labels.size), labels.astype(np.int64)] = 1.0
    return encoded


def build_nn1_art_classifier() -> tuple[object, int, str]:
    """Build ART PyTorchClassifier for NN1.

    Imports are intentionally inside this function so --help still works even
    before the attack dependencies are installed.
    """
    import torch
    import torch.nn as nn
    from art.estimators.classification import PyTorchClassifier
    from facenet_pytorch import InceptionResnetV1

    device_type = "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InceptionResnetV1(pretrained="vggface2", classify=True).eval().to(device)
    num_classes = getattr(getattr(model, "logits", None), "out_features", None)
    if num_classes is None:
        num_classes = 8631

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(3, 160, 160),
        nb_classes=num_classes,
        clip_values=(0.0, 1.0),
        channels_first=True,
        device_type=device_type,
    )
    return classifier, int(num_classes), device_type


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty manifest")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def progress_line(done: int, total: int, batch_index: int) -> str:
    percent = 100.0 * done / total
    return f"batch {batch_index}: {done}/{total} images ({percent:.1f}%)"


def default_output_dir(base_dir: Path, attack_name: str, eps: float) -> Path:
    eps_token = f"{eps:.3f}".replace(".", "_")
    return base_dir / attack_name / f"eps_{eps_token}"


def ensure_clean_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory already exists and is not empty: {output_dir}. "
            "Use --overwrite to allow writing into it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def finite_or_nan(value: int | float | None) -> int | float | str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value
