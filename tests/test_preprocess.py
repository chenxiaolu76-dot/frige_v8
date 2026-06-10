from __future__ import annotations

import numpy as np

from app.core.image_preprocess import (
    convert_gray,
    edge_detection,
    enhance_contrast,
    morphological_process,
    preprocess_pipeline,
    remove_noise,
)


def create_test_image() -> np.ndarray:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = [120, 180, 220]
    return image


def test_convert_gray_returns_2d_image():
    image = create_test_image()
    gray = convert_gray(image)

    assert gray.ndim == 2
    assert gray.shape == image.shape[:2]


def test_enhance_contrast_preserves_shape():
    image = create_test_image()
    enhanced = enhance_contrast(image)

    assert enhanced.shape == image.shape


def test_remove_noise_preserves_shape():
    image = create_test_image()
    denoised = remove_noise(image)

    assert denoised.shape == image.shape


def test_edge_detection_returns_2d_map():
    image = create_test_image()
    edges = edge_detection(image)

    assert edges.ndim == 2
    assert edges.shape == image.shape[:2]


def test_morphological_process_preserves_binary_shape():
    binary_image = np.zeros((64, 64), dtype=np.uint8)
    binary_image[20:40, 20:40] = 255
    processed = morphological_process(binary_image)

    assert processed.shape == binary_image.shape


def test_preprocess_pipeline_returns_expected_keys():
    image = create_test_image()
    results = preprocess_pipeline(
        image,
        use_gray=True,
        use_contrast=True,
        use_denoise=True,
        use_edge=True,
        use_morphology=True,
    )

    assert "original" in results
    assert "gray" in results
    assert "contrast" in results
    assert "denoised" in results
    assert "edges" in results
    assert "morphology" in results
    assert "final" in results
