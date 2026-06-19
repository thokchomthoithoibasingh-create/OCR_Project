"""
pdf_handler.py
--------------
Handles scanned PDF documents by converting each page into an image
(PaddleOCR cannot read PDFs directly — it only understands images),
running OCR on every page, and merging results back together in the
correct page order.

Uses pdf2image, which depends on the Poppler binaries being installed
and available on the system PATH (see PHASE 2 of the guide for the
Windows installation steps — this is the #1 source of beginner errors
in this whole project).
"""

import os
from pdf2image import convert_from_path
import numpy as np
import cv2


def pdf_to_images(pdf_path: str, dpi: int = 300, poppler_path: str = None) -> list:
    """
    Convert every page of a PDF into a list of in-memory images.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    dpi : int
        Resolution to render each page at. 300 DPI is the standard
        "print quality" resolution and gives PaddleOCR enough detail
        to read normal body text reliably. Going lower (150) is faster
        but can hurt accuracy on small fonts; going higher (600) is
        slower with usually little accuracy benefit.
    poppler_path : str, optional
        On Windows, pdf2image needs to know where Poppler's binaries
        (pdftoppm.exe etc.) live, UNLESS Poppler's bin folder is already
        on the system PATH. Pass the folder path here if you did not
        add Poppler to PATH during installation.

    Returns
    -------
    list of np.ndarray
        One BGR image array per PDF page, in original page order.

    Raises
    ------
    FileNotFoundError
        If the PDF does not exist.
    Exception
        Re-raised from pdf2image with a clarified message if Poppler
        is not found (the most common beginner error).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        if poppler_path:
            pil_pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        else:
            pil_pages = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        raise Exception(
            "Failed to convert PDF to images. This is almost always because "
            "Poppler is not installed or not on PATH.\n"
            "See PHASE 2 'Install Poppler' in the project guide.\n"
            f"Original error: {e}"
        )

    # Convert each PIL image (RGB) to an OpenCV-style NumPy array (BGR)
    cv_pages = []
    for pil_image in pil_pages:
        rgb_array = np.array(pil_image)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        cv_pages.append(bgr_array)

    return cv_pages


def save_page_images(pages: list, output_dir: str, base_filename: str) -> list:
    """
    Optionally save each converted page as a .png file to disk.
    Useful for debugging and for report screenshots showing
    "PDF page -> image -> OCR" pipeline stages.

    Returns the list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for i, page in enumerate(pages, start=1):
        path = os.path.join(output_dir, f"{base_filename}_page{i}.png")
        cv2.imwrite(path, page)
        saved_paths.append(path)
    return saved_paths
