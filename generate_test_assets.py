"""
generate_test_assets.py
------------------------
Generates synthetic test images and a multi-page test PDF so you have
something concrete to run OCR against immediately, without needing to
scan or photograph real documents first.

Run this ONCE after setup, from the project root:
    python tests/generate_test_assets.py

It creates:
    tests/test_images/clean_text.png       -> easy case, clean black text on white
    tests/test_images/noisy_text.png       -> harder case, text + visual noise
    tests/test_images/rotated_text.png     -> tests the angle classifier
    tests/test_images/low_res_text.png     -> tests the resize preprocessing step
    tests/test_pdfs/sample_multipage.pdf   -> 3-page PDF for PDF OCR testing

These are intentionally simple synthetic images (not real scanned
documents) — good enough to verify your pipeline WORKS end-to-end.
For your PHASE 7 accuracy report, you should also test with at least
one real scanned/photographed document for a realistic accuracy number.
"""

import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "tests", "test_images")
TEST_PDFS_DIR = os.path.join(PROJECT_ROOT, "tests", "test_pdfs")


def make_clean_text_image(path):
    img = np.ones((400, 900, 3), dtype=np.uint8) * 255
    lines = [
        "PaddleOCR Test Document",
        "This is a clean test image.",
        "Line three has Numbers 12345.",
        "Confidence should be HIGH here.",
    ]
    y = 70
    for line in lines:
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        y += 80
    cv2.imwrite(path, img)


def make_noisy_text_image(path):
    img = np.ones((400, 900, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Noisy Document Sample Text", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Testing denoise preprocessing step", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # Add synthetic salt-and-pepper noise to simulate a poor scan
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    noisy_img = cv2.subtract(img, noise)
    cv2.imwrite(path, noisy_img)


def make_rotated_text_image(path):
    img = np.ones((500, 900, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Rotated Angle Test Line", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # Rotate the whole canvas by 7 degrees to simulate a skewed scan
    center = (450, 250)
    rotation_matrix = cv2.getRotationMatrix2D(center, 7, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (900, 500), borderValue=(255, 255, 255))
    cv2.imwrite(path, rotated)


def make_low_res_text_image(path):
    # Deliberately tiny image to test the resize-upscale preprocessing step
    img = np.ones((60, 200, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Tiny Text", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def make_multipage_pdf(path):
    """
    Builds a simple 3-page PDF from generated images using Pillow,
    which is sufficient for testing the PDF->image->OCR pipeline
    without needing any scanner or external PDF tool.
    """
    from PIL import Image

    pages = []
    page_texts = [
        "PDF Page One - Introduction Section",
        "PDF Page Two - Methodology Section",
        "PDF Page Three - Conclusion Section",
    ]
    for text in page_texts:
        img = np.ones((600, 1000, 3), dtype=np.uint8) * 255
        cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "Sample body text for multi-page PDF OCR testing.", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pages.append(Image.fromarray(rgb))

    pages[0].save(path, save_all=True, append_images=pages[1:])


def main():
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(TEST_PDFS_DIR, exist_ok=True)

    print("[INFO] Generating test images...")
    make_clean_text_image(os.path.join(TEST_IMAGES_DIR, "clean_text.png"))
    make_noisy_text_image(os.path.join(TEST_IMAGES_DIR, "noisy_text.png"))
    make_rotated_text_image(os.path.join(TEST_IMAGES_DIR, "rotated_text.png"))
    make_low_res_text_image(os.path.join(TEST_IMAGES_DIR, "low_res_text.png"))
    print(f"[SUCCESS] 4 test images created in: {TEST_IMAGES_DIR}")

    print("[INFO] Generating multi-page test PDF...")
    try:
        make_multipage_pdf(os.path.join(TEST_PDFS_DIR, "sample_multipage.pdf"))
        print(f"[SUCCESS] Test PDF created in: {TEST_PDFS_DIR}")
    except ImportError:
        print("[WARNING] Pillow not installed — skipping PDF generation. Run: pip install Pillow")


if __name__ == "__main__":
    main()
