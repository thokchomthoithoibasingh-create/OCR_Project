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
from pdf2image.pdf2image import pdfinfo_from_path
import numpy as np
import cv2


def get_pdf_page_count(pdf_path: str, poppler_path: str = None) -> int:
    """
    Return the number of pages in a PDF WITHOUT rendering any of them to
    images. This is cheap (reads PDF metadata only) and lets the caller
    know how many pages exist before deciding how to process them.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path) if poppler_path \
            else pdfinfo_from_path(pdf_path)
        return int(info["Pages"])
    except Exception as e:
        raise Exception(
            "Failed to read PDF page count. This is almost always because "
            "Poppler is not installed or not on PATH.\n"
            "See PHASE 2 'Install Poppler' in the project guide.\n"
            f"Original error: {e}"
        )


def convert_pdf_page(pdf_path: str, page_number: int, dpi: int = 150, poppler_path: str = None) -> np.ndarray:
    """
    Convert a SINGLE PDF page to an in-memory BGR image, using
    first_page/last_page to render only that one page rather than the
    whole document.

    Why this exists: convert_from_path() without first_page/last_page
    renders every page into memory at once before returning anything.
    For long or large-paged PDFs, that can exhaust available RAM (seen
    as a MemoryError during page rendering) even though each individual
    page would have been fine on its own. Rendering one page at a time
    keeps memory use roughly constant regardless of document length.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    page_number : int
        1-indexed page number to convert.
    dpi : int
        Render resolution. See pdf_to_images() docstring for guidance.
    poppler_path : str, optional
        Folder containing Poppler's binaries, if not on system PATH.

    Returns
    -------
    np.ndarray
        BGR image array for the requested page.
    """
    try:
        kwargs = dict(dpi=dpi, first_page=page_number, last_page=page_number, thread_count=1)
        if poppler_path:
            kwargs["poppler_path"] = poppler_path
        pil_pages = convert_from_path(pdf_path, **kwargs)
    except MemoryError as e:
        raise Exception(
            f"Ran out of memory while rendering page {page_number}. Try a lower "
            f"dpi value (e.g. 100). Original error: {e}"
        )
    except Exception as e:
        raise Exception(
            f"Failed to convert page {page_number} to an image. "
            f"Original error: {e}"
        )

    if not pil_pages:
        raise Exception(f"Poppler returned no image for page {page_number}.")

    rgb_array = np.array(pil_pages[0])
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def iter_pdf_pages(pdf_path: str, dpi: int = 150, poppler_path: str = None):
    """
    Generator that yields (page_number, image) one page at a time for an
    entire PDF, WITHOUT ever holding more than one rendered page in
    memory at once. Prefer this over pdf_to_images() for long PDFs or on
    machines with limited RAM.

    Usage:
        for page_num, image in iter_pdf_pages(path):
            ... process image ...
            # image can be garbage collected after each loop iteration
    """
    total_pages = get_pdf_page_count(pdf_path, poppler_path=poppler_path)
    for page_number in range(1, total_pages + 1):
        image = convert_pdf_page(pdf_path, page_number, dpi=dpi, poppler_path=poppler_path)
        yield page_number, image
        # 'image' goes out of scope after each yield resumes and is
        # eligible for garbage collection before the next page is rendered.


def pdf_to_images(pdf_path: str, dpi: int = 300, poppler_path: str = None) -> list:
    """
    Convert every page of a PDF into a list of in-memory images, all at
    once. Kept for backward compatibility / small PDFs and direct testing.

    WARNING: for long or large-paged PDFs, this holds every rendered page
    in memory simultaneously and can raise MemoryError. For PDFs with
    many pages (10+), prefer iter_pdf_pages() instead, which renders and
    yields one page at a time.

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
            pil_pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path, thread_count=1)
        else:
            pil_pages = convert_from_path(pdf_path, dpi=dpi, thread_count=1)
    except MemoryError as e:
        raise Exception(
            "Ran out of memory while rendering this PDF to images. This usually "
            "means the PDF has very large page dimensions and/or the dpi setting "
            "is too high for that page size. Try lowering dpi (e.g. dpi=150 or "
            "dpi=100) in the pdf_to_images() call, and confirm the PDF's page "
            f"size with pdfinfo. Original error: {e}"
        )
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
