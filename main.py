"""
main.py
-------
Command-line entry point for the OCR project.

This is the file you actually RUN. It presents a menu, takes user
input, validates it, and calls into the other modules (preprocess,
ocr_engine, pdf_handler, output_writer) to do the real work.

Run it from the project root with:
    python src/main.py

WHY a single CLI entry point:
Keeping all user interaction (menus, input(), print()) in this one
file — and keeping all the "real logic" in the other modules — means
the OCR/preprocessing code stays reusable and testable independently
of the command-line interface around it.
"""

import os
import sys
import glob
import traceback

# Allow running this file directly (python src/main.py) by adding the
# project root to sys.path, so "from src.xxx import yyy" style imports
# aren't required — we just import sibling modules directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_image
from ocr_engine import OCREngine
from pdf_handler import pdf_to_images
from output_writer import (
    format_results_for_display,
    save_image_ocr_output,
    save_pdf_ocr_output,
    calculate_average_confidence,
)

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "text_outputs")

# Set this if Poppler is NOT on your system PATH. Example:
# POPPLER_PATH = r"C:\poppler\Library\bin"
POPPLER_PATH = None


def print_banner():
    print("=" * 60)
    print("        AI-BASED OCR SYSTEM (PaddleOCR + Python)")
    print("=" * 60)


def print_menu():
    print("\nMAIN MENU")
    print("-" * 30)
    print("1. Extract text from an Image")
    print("2. Extract text from a scanned PDF")
    print("3. Batch process a folder of images")
    print("4. Exit")
    print("-" * 30)


def get_validated_path(prompt: str, must_exist: bool = True) -> str:
    """
    Prompt the user for a file path and validate it. Loops until a
    valid path is given, or the user types 'cancel' to abort.
    """
    while True:
        path = input(prompt).strip().strip('"')  # strip quotes pasted from Windows Explorer
        if path.lower() == "cancel":
            return None
        if must_exist and not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            print("        Tip: you can drag-and-drop the file into the terminal to paste its path.")
            continue
        return path


def handle_image_ocr(engine: OCREngine):
    print("\n--- IMAGE OCR ---")
    image_path = get_validated_path("Enter path to image file (or 'cancel'): ")
    if image_path is None:
        print("Cancelled.")
        return

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        print(f"[ERROR] Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}")
        return

    try:
        print("[INFO] Preprocessing image...")
        processed = preprocess_image(image_path, do_grayscale=True, do_denoise=True, do_resize=True)

        print("[INFO] Running OCR (this may take a few seconds)...")
        results = engine.run(processed)

        print("\n--- EXTRACTED TEXT ---")
        print(format_results_for_display(results, show_confidence=True))
        print("-" * 40)
        print(f"Average Confidence: {calculate_average_confidence(results)}%")
        print(f"Lines Detected    : {len(results)}")

        if not results:
            print("[WARNING] No text was detected. Try a clearer image or check preprocessing settings.")
            return

        output_path = save_image_ocr_output(results, image_path, OUTPUT_DIR)
        print(f"\n[SUCCESS] Output saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except ValueError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error during image OCR: {e}")
        traceback.print_exc()


def handle_pdf_ocr(engine: OCREngine):
    print("\n--- PDF OCR ---")
    pdf_path = get_validated_path("Enter path to PDF file (or 'cancel'): ")
    if pdf_path is None:
        print("Cancelled.")
        return

    if os.path.splitext(pdf_path)[1].lower() != ".pdf":
        print("[ERROR] File must have a .pdf extension.")
        return

    try:
        print("[INFO] Converting PDF pages to images...")
        pages = pdf_to_images(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        print(f"[INFO] {len(pages)} page(s) found.")

        all_page_results = []
        for i, page_image in enumerate(pages, start=1):
            print(f"[INFO] Running OCR on page {i}/{len(pages)}...")

            # Re-use preprocessing on each rendered page image.
            # We resize in-memory since the page is already a NumPy array.
            import cv2
            gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

            results = engine.run(processed)
            all_page_results.append(results)

            page_conf = calculate_average_confidence(results)
            print(f"         -> {len(results)} line(s) detected, avg confidence {page_conf}%")

        output_path = save_pdf_ocr_output(all_page_results, pdf_path, OUTPUT_DIR)

        print("\n--- SUMMARY ---")
        total_lines = sum(len(p) for p in all_page_results)
        print(f"Total Pages   : {len(pages)}")
        print(f"Total Lines   : {total_lines}")
        print(f"[SUCCESS] Output saved to: {output_path}")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error during PDF OCR: {e}")
        traceback.print_exc()


def handle_batch_ocr(engine: OCREngine):
    print("\n--- BATCH IMAGE OCR ---")
    folder_path = get_validated_path("Enter path to folder containing images (or 'cancel'): ")
    if folder_path is None:
        print("Cancelled.")
        return

    if not os.path.isdir(folder_path):
        print("[ERROR] That path is not a folder.")
        return

    image_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))

    if not image_files:
        print(f"[WARNING] No supported images found in {folder_path}")
        return

    print(f"[INFO] Found {len(image_files)} image(s). Starting batch processing...\n")

    success_count = 0
    fail_count = 0

    for idx, image_path in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        try:
            processed = preprocess_image(image_path, do_grayscale=True, do_denoise=True, do_resize=True)
            results = engine.run(processed)
            output_path = save_image_ocr_output(results, image_path, OUTPUT_DIR)
            conf = calculate_average_confidence(results)
            print(f"         -> {len(results)} line(s), avg confidence {conf}% -> saved: {os.path.basename(output_path)}")
            success_count += 1
        except Exception as e:
            print(f"         -> [ERROR] Failed: {e}")
            fail_count += 1

    print(f"\n[BATCH COMPLETE] Success: {success_count} | Failed: {fail_count}")


def main():
    print_banner()
    print("[INFO] Initializing OCR engine (loading PaddleOCR models)...")
    try:
        engine = OCREngine(lang="en", use_angle_cls=True)
    except Exception as e:
        print(f"[FATAL] Could not initialize PaddleOCR: {e}")
        print("        Check your installation — see PHASE 2 of the project guide.")
        sys.exit(1)

    while True:
        print_menu()
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            handle_image_ocr(engine)
        elif choice == "2":
            handle_pdf_ocr(engine)
        elif choice == "3":
            handle_batch_ocr(engine)
        elif choice == "4":
            print("\nExiting. Goodbye!")
            break
        else:
            print("[ERROR] Invalid choice. Please enter a number between 1 and 4.")


if __name__ == "__main__":
    main()
