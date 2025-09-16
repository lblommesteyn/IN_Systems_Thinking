from dataclasses import dataclass
from typing import List, Dict, Optional

import fitz  # PyMuPDF


@dataclass
class ExtractionConfig:
    min_chars: int = 50
    min_words: int = 10
    density_threshold: float = 0.0008  # chars per square point


class Extractor:
    """
    Extract text per page using native PDF text when available;
    fall back to OCR for image-only or low-density pages.
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()

    def _is_low_density(self, text: str, page: fitz.Page) -> bool:
        text = (text or "").strip()
        char_count = len(text)
        word_count = len(text.split())
        if char_count < self.config.min_chars or word_count < self.config.min_words:
            return True
        # density heuristic
        area = page.rect.width * page.rect.height
        if area > 0 and (char_count / area) < self.config.density_threshold:
            return True
        return False

    def extract_pages(self, doc: fitz.Document, ocr=None) -> List[Dict]:
        """
        Returns a list of dicts with keys: page, text, method, char_count
        If OCR object provided, use it as fallback when low density.
        """
        results: List[Dict] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            method = "text"
            if self._is_low_density(text, page):
                if ocr is None:
                    # Keep the original (possibly empty) text but mark low_density
                    results.append({
                        "page": i,
                        "text": text.strip(),
                        "method": method,
                        "char_count": len(text.strip()),
                        "low_density": True,
                    })
                    continue
                # Fallback to OCR
                ocr_text = ocr.page_to_text(page) or ""
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    method = "ocr"
            results.append({
                "page": i,
                "text": text.strip(),
                "method": method,
                "char_count": len(text.strip()),
                "low_density": self._is_low_density(text, page),
            })
        return results
