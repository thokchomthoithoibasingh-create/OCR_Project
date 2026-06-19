"""
preprocess.py
--------------
Image preprocessing utilities for the OCR pipeline.

WHY PREPROCESSING MATTERS:
PaddleOCR's text detector and recognizer were trained on a mix of clean and
noisy images, but OCR accuracy almost always improves when you hand the
model a clean, high-contrast, correctly-sized image instead of a raw photo
or scan. This module applies four classic preprocessing steps:

    1. Grayscale conversion -> removes color noise, OCR only needs intensity
    2. Denoising            -> removes scanner/camera sensor noise
    3. Thresholding         -> increases text/background contrast
    4. Resizing             -> upscales small text so the detector can find it

Each step is optional and controlled by flags so you can A/B test which
combination works best on your own sample images (this matters for your
PHASE 7 testing/accuracy report).
"""

import cv2
import numpy as np
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    Parameters
    ----------
    image_path : str
        Path to the image file (.jpg, .jpeg, .png, .bmp, .tiff)

    Returns
    -------
    np.ndarray
        Image in BGR format (OpenCV's default).

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If OpenCV could not decode the file (corrupt or unsupported format).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"OpenCV could not read this file as an image: {image_path}\n"
            f"It may be corrupted or in an unsupported format."
        )
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_image(gray_image: np.ndarray) -> np.ndarray:
    """
    Remove noise using OpenCV's fast non-local means denoising.

    fastNlMeansDenoising is preferred over a simple blur because it
    smooths noise while preserving text edges, which matters a lot
    for OCR accuracy on scanned/photographed documents.
    """
    return cv2.fastNlMeansDenoising(gray_image, h=10, templateWindowSize=7, searchWindowSize=21)


def apply_threshold(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding to binarize the image (pure black/white).

    Adaptive thresholding is used instead of a single global threshold
    because lighting is rarely uniform across a scanned page/photo —
    adaptive thresholding calculates a local threshold for each region.
    """
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )


def resize_image(image: np.ndarray, max_dimension: int = 2000) -> np.ndarray:
    """
    Resize the image so its longest side does not exceed max_dimension,
    upscaling small images and downscaling very large ones.

    Why: PaddleOCR's text detector works on a fixed internal resolution.
    Very small images make tiny text undetectable; very large images
    slow down inference without adding accuracy. 2000px is a reasonable
    practical ceiling for document images on a CPU.
    """
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side == max_dimension:
        return image

    scale = max_dimension / longest_side
    new_width = int(width * scale)
    new_height = int(height * scale)

    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def preprocess_image(
    image_path: str,
    do_grayscale: bool = True,
    do_denoise: bool = True,
    do_threshold: bool = False,
    do_resize: bool = True,
    save_debug_path: str = None,
) -> np.ndarray:
    """
    Run the full preprocessing pipeline on an image file and return a
    NumPy array ready to feed into PaddleOCR.

    NOTE on do_threshold default = False:
    Hard binary thresholding can sometimes *hurt* PaddleOCR's accuracy
    because the model was trained on natural color/grayscale images, not
    pure black/white. It is included and exposed as a flag (required by
    the project spec) but kept off by default. Encourage testing both
    ways in PHASE 7 and reporting which performed better on your samples.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    do_grayscale, do_denoise, do_threshold, do_resize : bool
        Toggle individual preprocessing steps.
    save_debug_path : str, optional
        If provided, saves the preprocessed image to this path so you
        can visually inspect what PaddleOCR actually "saw" — useful for
        screenshots in your report.

    Returns
    -------
    np.ndarray
        Preprocessed image array.
    """
    image = load_image(image_path)

    if do_resize:
        image = resize_image(image)

    if do_grayscale:
        image = convert_to_grayscale(image)

        if do_denoise:
            image = denoise_image(image)

        if do_threshold:
            image = apply_threshold(image)

        # PaddleOCR expects a 3-channel image even if visually grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif do_denoise:
        # Denoise in color if grayscale was skipped
        image = cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)

    if save_debug_path:
        cv2.imwrite(save_debug_path, image)

    return image
