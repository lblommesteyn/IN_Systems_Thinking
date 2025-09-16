import os
import io
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


@dataclass
class OCRConfig:
    lang: str = "eng"
    dpi: int = 300
    tesseract_cmd: Optional[str] = None  # full path to tesseract executable (Windows)


class OCR:
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        # Configure tesseract command if provided or via env var
        cmd = self.config.tesseract_cmd or os.getenv("TESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        else:
            # Windows default install path (best-effort). Ignore if not present.
            default_win = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            if os.name == "nt" and os.path.exists(default_win):
                pytesseract.pytesseract.tesseract_cmd = default_win

    def page_to_text(self, page: fitz.Page) -> str:
        """Render a page to an image and OCR it."""
        scale = self.config.dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang=self.config.lang)
        return text
