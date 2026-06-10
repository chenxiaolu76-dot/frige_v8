from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_image(image_source: str | Path | Any) -> np.ndarray:
    """
    Read an image from a local path or a Streamlit uploaded file.

    Input:
        image_source: Local image path or uploaded file object.

    Output:
        Loaded BGR image as a numpy array.
    """
    if isinstance(image_source, (str, Path)):
        image = cv2.imread(str(image_source))
        if image is None:
            raise ValueError(f"Failed to read image from path: {image_source}")
        return image

    if hasattr(image_source, "read"):
        file_bytes = np.asarray(bytearray(image_source.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode uploaded image.")
        return image

    raise TypeError("image_source must be a file path or an uploaded file object.")


def convert_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.

    Input:
        image: BGR image as a numpy array.

    Output:
        Grayscale image as a numpy array.
    """
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.

    Input:
        image: Grayscale or BGR image as a numpy array.
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid size.

    Output:
        Contrast-enhanced image as a numpy array.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:
        return clahe.apply(image)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def remove_noise(image: np.ndarray, gaussian_kernel: tuple[int, int] = (5, 5), median_kernel: int = 5) -> np.ndarray:
    """
    Remove image noise using Gaussian blur and median blur.

    Input:
        image: Input image as a numpy array.
        gaussian_kernel: Gaussian blur kernel size.
        median_kernel: Median blur kernel size.

    Output:
        Denoised image as a numpy array.
    """
    gaussian_blurred = cv2.GaussianBlur(image, gaussian_kernel, 0)
    return cv2.medianBlur(gaussian_blurred, median_kernel)


def edge_detection(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges using the Canny algorithm.

    Input:
        image: Grayscale or BGR image as a numpy array.
        low_threshold: Lower Canny threshold.
        high_threshold: Upper Canny threshold.

    Output:
        Edge map as a numpy array.
    """
    gray_image = convert_gray(image)
    return cv2.Canny(gray_image, low_threshold, high_threshold)


def morphological_process(
    image: np.ndarray,
    kernel_size: tuple[int, int] = (3, 3),
    open_iterations: int = 1,
    close_iterations: int = 1,
) -> np.ndarray:
    """
    Apply morphological opening and closing operations.

    Input:
        image: Binary or grayscale image as a numpy array.
        kernel_size: Structuring element size.
        open_iterations: Number of opening iterations.
        close_iterations: Number of closing iterations.

    Output:
        Morphologically processed image as a numpy array.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    return closed


def preprocess_pipeline(
    image: np.ndarray,
    use_gray: bool = False,
    use_contrast: bool = True,
    use_denoise: bool = True,
    use_edge: bool = False,
    use_morphology: bool = False,
) -> dict[str, np.ndarray]:
    """
    Run a configurable preprocessing pipeline.

    Input:
        image: Input image as a numpy array.
        use_gray: Whether to convert to grayscale.
        use_contrast: Whether to apply CLAHE contrast enhancement.
        use_denoise: Whether to apply denoising.
        use_edge: Whether to run edge detection.
        use_morphology: Whether to apply morphological operations.

    Output:
        A dictionary containing intermediate and final processed images.
    """
    results: dict[str, np.ndarray] = {"original": image.copy()}
    current_image = image.copy()

    if use_gray:
        current_image = convert_gray(current_image)
        results["gray"] = current_image.copy()

    if use_contrast:
        current_image = enhance_contrast(current_image)
        results["contrast"] = current_image.copy()

    if use_denoise:
        current_image = remove_noise(current_image)
        results["denoised"] = current_image.copy()

    if use_edge:
        current_image = edge_detection(current_image)
        results["edges"] = current_image.copy()

    if use_morphology:
        current_image = morphological_process(current_image)
        results["morphology"] = current_image.copy()

    results["final"] = current_image
    return results


if __name__ == "__main__":
    sample_path = Path("data/samples/sample_fridge.jpg")

    if sample_path.exists():
        sample_image = load_image(sample_path)
        pipeline_results = preprocess_pipeline(
            sample_image,
            use_gray=False,
            use_contrast=True,
            use_denoise=True,
            use_edge=True,
            use_morphology=True,
        )

        output_dir = Path("data/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, result_image in pipeline_results.items():
            output_path = output_dir / f"{name}.png"
            cv2.imwrite(str(output_path), result_image)

        print("Preprocessing demo finished. Results saved to data/outputs/")
    else:
        print("Sample image not found: data/samples/sample_fridge.jpg")
