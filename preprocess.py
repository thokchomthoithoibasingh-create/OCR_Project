import cv2
import numpy as np
import os


def load_image(image_path: str) -> np.ndarray:
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
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_image(gray_image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray_image, h=10, templateWindowSize=7, searchWindowSize=21)


def apply_threshold(gray_image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )


def resize_image(image: np.ndarray, max_dimension: int = 2000) -> np.ndarray:
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side == max_dimension:
        return image

    scale = max_dimension / longest_side
    new_width = int(width * scale)
    new_height = int(height * scale)

    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def preprocess_array(
    image: np.ndarray,
    do_grayscale: bool = True,
    do_denoise: bool = True,
    do_threshold: bool = False,
    do_resize: bool = True,
    save_debug_path: str = None,
) -> np.ndarray:
    """
    Run the full preprocessing pipeline on an already-loaded image array
    (e.g. a PDF page rasterized by pdf_handler.iter_pdf_pages, or any
    in-memory frame) and return a NumPy array ready to feed into
    PaddleOCR.

    This is the shared core used by both `preprocess_image` (file-path
    entry point, for the image-upload flow) and main.py's PDF flow, so
    the two flows can never drift out of sync on denoise parameters,
    thresholding availability, or the grayscale->3-channel conversion
    PaddleOCR requires.

    See `preprocess_image` for parameter docs — identical, minus the
    file-loading step.
    """
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
    return preprocess_array(
        image,
        do_grayscale=do_grayscale,
        do_denoise=do_denoise,
        do_threshold=do_threshold,
        do_resize=do_resize,
        save_debug_path=save_debug_path,
    )