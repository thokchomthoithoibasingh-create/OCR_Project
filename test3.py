"""
main.py
-------
Command-line entry point for the OCR project.
Run: python main.py
"""

import os
import sys
import glob
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_image
from ocr_engine import OCREngine
from pdf_handler import iter_pdf_pages, get_pdf_page_count
from output_writer import (
    format_results_for_display,
    save_image_ocr_output,
    save_pdf_ocr_output,
    calculate_average_confidence,
)

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# Fixed output path — saves inside your project folder
OUTPUT_DIR = "/workspaces/OCR_Project/output/text_outputs"

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
    while True:
        path = input(prompt).strip().strip('"')
        if path.lower() == "cancel":
            return None
        if must_exist and not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            print("        Tip: drag-and-drop the file into the terminal to paste its path.")
            continue
        return path


def show_txt_output(txt_path: str):
    print("\n" + "=" * 60)
    print("               TXT OUTPUT")
    print("=" * 60)
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as e:
        print(f"[ERROR] Could not read txt file: {e}")
    print("=" * 60)
    print(f"[TXT FILE] Saved at: {txt_path}")
    print("=" * 60)


def show_docx_output(docx_path: str):
    if docx_path is None:
        print("\n[INFO] DOCX skipped — run: pip install python-docx")
        return
    print("\n" + "=" * 60)
    print("               DOCX OUTPUT")
    print("=" * 60)
    try:
        from docx import Document
        doc = Document(docx_path)
        for para in doc.paragraphs:
            if para.text.strip():
                print(para.text)
    except Exception as e:
        print(f"[ERROR] Could not read docx file: {e}")
    print("=" * 60)
    print(f"[DOCX FILE] Saved at: {docx_path}")
    print("=" * 60)


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

        if not results:
            print("[WARNING] No text was detected.")
            return

        print(f"\nAverage Confidence: {calculate_average_confidence(results)}%")
        print(f"Lines Detected    : {len(results)}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        txt_path, docx_path = save_image_ocr_output(results, image_path, OUTPUT_DIR)

        print(f"\n[SUCCESS] TXT  saved to: {txt_path}")
        if docx_path:
            print(f"[SUCCESS] DOCX saved to: {docx_path}")

        show_txt_output(txt_path)
        show_docx_output(docx_path)

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
        import cv2
        from preprocess import resize_image

        print("[INFO] Reading PDF page count...")
        total_pages = get_pdf_page_count(pdf_path, poppler_path=POPPLER_PATH)
        print(f"[INFO] {total_pages} page(s) found.")

        all_page_results = []
        for page_number, page_image in iter_pdf_pages(pdf_path, dpi=150, poppler_path=POPPLER_PATH):
            print(f"[INFO] Running OCR on page {page_number}/{total_pages}...", flush=True)

            resized = resize_image(page_image, max_dimension=2000)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            try:
                denoised = cv2.fastNlMeansDenoising(gray, h=10)
            except cv2.error as denoise_err:
                print(f"         -> [WARNING] Denoising failed ({denoise_err}); using grayscale.")
                denoised = gray

            processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            results = engine.run(processed)
            all_page_results.append(results)

            page_conf = calculate_average_confidence(results)
            print(f"         -> {len(results)} line(s) detected, avg confidence {page_conf}%")
            del page_image, resized, gray, denoised, processed

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        txt_path, docx_path = save_pdf_ocr_output(all_page_results, pdf_path, OUTPUT_DIR)

        total_lines = sum(len(p) for p in all_page_results)
        print("\n--- SUMMARY ---")
        print(f"Total Pages   : {total_pages}")
        print(f"Total Lines   : {total_lines}")
        print(f"\n[SUCCESS] TXT  saved to: {txt_path}")
        if docx_path:
            print(f"[SUCCESS] DOCX saved to: {docx_path}")

        show_txt_output(txt_path)
        show_docx_output(docx_path)

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
    saved_pairs = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, image_path in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        try:
            processed = preprocess_image(image_path, do_grayscale=True, do_denoise=True, do_resize=True)
            results = engine.run(processed)
            txt_path, docx_path = save_image_ocr_output(results, image_path, OUTPUT_DIR)
            conf = calculate_average_confidence(results)
            print(f"         -> {len(results)} line(s), avg confidence {conf}%")
            print(f"            TXT : {txt_path}")
            if docx_path:
                print(f"            DOCX: {docx_path}")
            saved_pairs.append((txt_path, docx_path))
            success_count += 1
        except Exception as e:
            print(f"         -> [ERROR] Failed: {e}")
            fail_count += 1

    print(f"\n[BATCH COMPLETE] Success: {success_count} | Failed: {fail_count}")

    for txt_path, docx_path in saved_pairs:
        show_txt_output(txt_path)
        show_docx_output(docx_path)


def main():
    print_banner()
    print(f"[INFO] Output folder: {OUTPUT_DIR}")
    print("[INFO] Initializing OCR engine (loading PaddleOCR models)...")
    try:
        engine = OCREngine(lang="en", use_angle_cls=True)
    except Exception as e:
        print(f"[FATAL] Could not initialize PaddleOCR: {e}")
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