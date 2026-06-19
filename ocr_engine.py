"""
ocr_engine.py
-------------
Wraps PaddleOCR so the rest of the project never talks to the PaddleOCR
API directly. This is a deliberate design choice: if PaddleOCR's API
changes between versions (it has, historically), you only fix it in
ONE place.

This module is responsible for:
    - Initializing the PaddleOCR model (once, and re-using it)
    - Running OCR on a single preprocessed image (NumPy array)
    - Returning a clean, consistent result structure:
        [
            {
                "text": "Hello World",
                "confidence": 0.987,
                "box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            },
            ...
        ]
    - Sorting results top-to-bottom, left-to-right to preserve reading
      order/alignment as closely as possible in plain text output.
"""

from paddleocr import PaddleOCR
import numpy as np


class OCREngine:
    """
    Thin wrapper around PaddleOCR.

    PaddleOCR model initialization is slow (it loads detection +
    recognition + angle-classification neural networks). We initialize
    ONCE per program run and reuse the same instance for every image —
    this is why OCREngine is a class and not a function.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = True):
        """
        Parameters
        ----------
        lang : str
            Language model to load. 'en' = English. PaddleOCR supports
            ~80 languages; change this single value to add another
            language later (e.g. 'hi' for Hindi, 'ch' for Chinese).
        use_angle_cls : bool
            Whether to run the text-angle classifier, which detects and
            corrects rotated/upside-down text. Recommended True for
            scanned documents and photographed pages.
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.ocr = self._build_engine()

    def _build_engine(self) -> PaddleOCR:
        """
        Construct a fresh PaddleOCR instance.

        Pulled out into its own method (rather than inlined in __init__)
        so that run() can call it again to rebuild the engine if
        PaddleOCR's CPU backend gets into a bad state mid-run (see
        run()'s retry logic below).

        enable_mkldnn=False: PaddleOCR's MKLDNN (oneDNN) backend on CPU
        has a known bug where its internal primitive cache can become
        corrupted after a number of inference calls — especially when
        input image dimensions vary between calls, as they do across
        PDF pages. This throws "RuntimeError: could not create a
        primitive" at an unpredictable point (e.g. page 5 one run,
        page 7 the next). Disabling MKLDNN avoids the corruption
        entirely; it costs some CPU speed but is far more reliable for
        long batch/PDF runs.
        """
        print(f"[INFO] Loading PaddleOCR model (lang='{self.lang}') — this can take 10-30s on first run...")
        engine = PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=self.lang,
            enable_mkldnn=False,
            show_log=False,
        )
        print("[INFO] PaddleOCR model loaded successfully.")
        return engine

    def run(self, image: np.ndarray, _retried: bool = False) -> list:
        """
        Run OCR on a single preprocessed image array.

        Parameters
        ----------
        image : np.ndarray
            BGR image array (e.g. from preprocess.preprocess_image).
        _retried : bool
            Internal flag — True when this call is itself a retry, so we
            don't recurse forever if the engine rebuild doesn't help.

        Returns
        -------
        list of dict
            Each dict has keys: 'text', 'confidence', 'box'.
            Sorted in approximate reading order (top-to-bottom,
            left-to-right) to best preserve document layout in plain text.
        """
        try:
            raw_result = self.ocr.ocr(image, cls=True)
        except RuntimeError as e:
            # Belt-and-braces: even with enable_mkldnn=False this backend
            # can occasionally throw a transient RuntimeError on CPU.
            # Rebuild the underlying PaddleOCR instance once and retry
            # this same image before giving up.
            if _retried:
                raise  # already retried once — let the caller handle it
            print(f"[WARNING] OCR backend error ({e}); reinitializing engine and retrying this image once...")
            self.ocr = self._build_engine()
            return self.run(image, _retried=True)

        # PaddleOCR returns a list with one element per input image.
        # Since we pass one image, we take index [0]. It can be None
        # if absolutely no text was detected.
        if not raw_result or raw_result[0] is None:
            return []

        page_result = raw_result[0]

        extracted = []
        for line in page_result:
            box = line[0]                 # 4 corner points
            text, confidence = line[1]    # (text_string, confidence_float)
            extracted.append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "box": box,
            })

        return self._sort_reading_order(extracted)

    @staticmethod
    def _sort_reading_order(results: list) -> list:
        """
        Sort detected text lines into approximate natural reading order:
        top-to-bottom first, then left-to-right within the same line band.

        WHY: PaddleOCR returns results roughly in detection order, which
        does not always match how a human reads the page (especially for
        multi-column documents or scattered text). We approximate reading
        order by:
            1. Computing the vertical center (y) of each box
            2. Grouping lines whose y-centers are close together (same row)
            3. Sorting each row left-to-right by x

        This is a heuristic, not a perfect layout engine — true multi-column
        layout reconstruction would need a proper document-layout model.
        """
        if not results:
            return results

        def y_center(item):
            ys = [point[1] for point in item["box"]]
            return sum(ys) / len(ys)

        def x_left(item):
            xs = [point[0] for point in item["box"]]
            return min(xs)

        # Sort everything by vertical position first
        results_sorted = sorted(results, key=y_center)

        # Group into rows: lines whose y-centers fall within a tolerance
        # band are treated as being on the same visual line.
        row_tolerance = 15  # pixels; tuned for typical document scans
        rows = []
        current_row = [results_sorted[0]]
        current_y = y_center(results_sorted[0])

        for item in results_sorted[1:]:
            if abs(y_center(item) - current_y) <= row_tolerance:
                current_row.append(item)
            else:
                rows.append(current_row)
                current_row = [item]
                current_y = y_center(item)
        rows.append(current_row)

        # Sort each row left-to-right, then flatten
        final_order = []
        for row in rows:
            row_sorted = sorted(row, key=x_left)
            final_order.extend(row_sorted)

        return final_order