OCR_Project

OCR_Project is a lightweight OCR processing toolkit focused on extracting text and simple table structures from image and PDF inputs. It provides preprocessing, PDF handling, table detection, and flexible output writing to text files.

**Key features**
- Preprocessing pipeline for image cleanup and enhancement
- PDF handling and image extraction
- OCR text extraction and simple table detection
- CLI and Streamlit demo app for interactive use
- Simple input/output folder conventions for batch processing

**Requirements**
- Python 3.8+
- See `requirements.txt` for full dependency list

Installation
------------

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Running the project
-------------------

There are a few entry points depending on the workflow you want:

- Batch CLI (basic):

```bash
python main.py
```

- Streamlit demo (interactive):

```bash
streamlit run streamlit_app.py
```

- Generate sample test assets (helper script):

```bash
python generate_test_assets.py
```

Project layout
--------------

- `input/` — put your input images or PDFs here.
- `output/` — generated outputs and text files are written here.
- `ocr_engine.py` — main OCR and text extraction logic.
- `preprocess.py` — image preprocessing utilities.
- `pdf_handler.py` — PDF-to-image conversion and handling.
- `table_detector.py` — simple heuristics for detecting tables.
- `output_writer.py` — formatting and writing extracted text to files.
- `streamlit_app.py` — interactive demo UI.
- `generate_test_assets.py` — creates sample inputs for testing.

Usage notes
-----------

- Place input files in `input/` and run `python main.py` to process them in batch; results will be placed under `output/text_outputs/`.
- The Streamlit app exposes the same processing pipeline for trying different preprocessing options and inspecting results.

Testing
-------

There is no automated test suite included; to manually test, run the generator and then process the generated assets:

```bash
python generate_test_assets.py
python main.py
```

Windows
-------

On Windows you can use the provided `run.bat` to run the main pipeline.

Contributing
------------

Contributions are welcome. Open an issue or a PR describing the change you propose. Keep changes focused and include small, testable commits.

License
-------

This repository does not include a license file. If you plan to publish or share this project, add a `LICENSE` file to clarify terms.

Contact
-------

If you have questions about the code, open an issue in this repository.
