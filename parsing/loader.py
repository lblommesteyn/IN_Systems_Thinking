import os
from typing import Iterator, Tuple
import fitz  # PyMuPDF


class Loader:
    """Open documents and provide page iteration utilities."""

    def open_pdf(self, file_path: str) -> fitz.Document:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            return fitz.open(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {file_path}: {e}")

    def iter_pages(self, doc: fitz.Document) -> Iterator[Tuple[int, fitz.Page]]:
        """Yield (page_number, page) for each page in the document."""
        for i in range(doc.page_count):
            yield i, doc.load_page(i)
