"""
streamlit_app.py
-----------------
Simple web UI for the OCR pipeline, built on top of the existing
ocr_engine.py / output_writer.py / preprocess.py modules - no logic
is duplicated, this just wraps them with a UI.

Run with:
    streamlit run streamlit_app.py
"""

import os
import tempfile
import streamlit as st

from preprocess import preprocess_image
from ocr_engine import OCREngine
from pdf_handler import pdf_to_images
from output_writer import (
    format_plain_text,
    calculate_average_confidence,
    save_image_ocr_output,
    save_pdf_ocr_output,
)

SUPPORTED_LANGUAGES = {"English": "en", "Hindi": "hi", "Chinese": "ch", "French": "fr", "Spanish": "es", "Arabic": "ar"}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "text_outputs")

st.set_page_config(page_title="AI OCR System", layout="wide")
st.title("AI-Based OCR System")

lang_name = st.sidebar.selectbox("OCR Language", list(SUPPORTED_LANGUAGES.keys()))
lang_code = SUPPORTED_LANGUAGES[lang_name]

@st.cache_resource
def load_engine(lang):
    return OCREngine(lang=lang, use_angle_cls=True)

uploaded_file = st.file_uploader("Upload an image or scanned PDF", type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    engine = load_engine(lang_code)

    if uploaded_file.name.lower().endswith(".pdf"):
        with st.spinner("Converting PDF pages and running OCR..."):
            pages = pdf_to_images(tmp_path)
            all_results = [engine.run(p) for p in pages]
        full_text = "\n\n".join(format_plain_text(r) for r in all_results)
        txt_path, docx_path = save_pdf_ocr_output(all_results, uploaded_file.name, OUTPUT_DIR)
        avg_conf = calculate_average_confidence([item for page in all_results for item in page])
    else:
        with st.spinner("Preprocessing and running OCR..."):
            processed = preprocess_image(tmp_path, do_grayscale=True, do_denoise=True, do_resize=True)
            results = engine.run(processed)
        full_text = format_plain_text(results)
        txt_path, docx_path = save_image_ocr_output(results, uploaded_file.name, OUTPUT_DIR)
        avg_conf = calculate_average_confidence(results)

    st.subheader("Extracted Text")
    st.text_area("Result", full_text, height=400)
    st.metric("Average Confidence", f"{avg_conf}%")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download TXT", open(txt_path, "rb"), file_name=os.path.basename(txt_path))
    with col2:
        if docx_path:
            st.download_button("Download DOCX", open(docx_path, "rb"), file_name=os.path.basename(docx_path))

    os.remove(tmp_path)