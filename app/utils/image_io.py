from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_image_from_path(image_path: str | Path) -> np.ndarray:
    """Load an image from a local file path."""
    file_path = Path(image_path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError(f"Failed to load image from path: {file_path}")
    return image


def load_image_from_upload(uploaded_file: Any) -> np.ndarray:
    """Load an image from a Streamlit uploaded file object."""
    if not hasattr(uploaded_file, "read"):
        raise TypeError("uploaded_file must provide a read() method.")

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode uploaded image.")
    return image


def save_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save an image to disk and return the resolved path."""
    file_path = Path(output_path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(file_path), image)
    if not success:
        raise ValueError(f"Failed to save image to path: {file_path}")
    return file_path


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image_keep_ratio(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resize an image while keeping its aspect ratio."""
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy()

    scale = max_width / width
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
