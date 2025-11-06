from typing import List, Dict, Optional

try:
    # Prefer existing robust segmenter
    from src.preprocessing.document_parser import DocumentParser  # type: ignore
except Exception:  # pragma: no cover - fallback
    DocumentParser = None  # type: ignore


class Paragraphizer:
    """Segments page texts into paragraphs per page using existing rules."""

    def __init__(self):
        self._dp = DocumentParser() if DocumentParser else None

    def segment_text(self, text: str) -> List[Dict[str, str]]:
        if self._dp:
            return self._dp.segment_paragraphs(text)
        # Simple fallback: split on blank lines
        paras: List[Dict[str, str]] = []
        buf: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                if buf:
                    joined = " ".join(buf).strip()
                    if 50 <= len(joined) <= 2000:
                        paras.append({"text": joined})
                    buf = []
            else:
                buf.append(line)
        if buf:
            joined = " ".join(buf).strip()
            if 50 <= len(joined) <= 2000:
                paras.append({"text": joined})
        return paras

    def paragraphs_from_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Input: list of {page, text, method, ...}
        Output: list of paragraphs with metadata: {text, page, method, para_index}
        """
        out: List[Dict] = []
        for page_entry in pages:
            page_num = page_entry["page"]
            method = page_entry.get("method", "text")
            text = page_entry.get("text", "")
            paras = self.segment_text(text)
            for pi, p in enumerate(paras):
                out.append({
                    "text": p["text"],
                    "page": page_num,
                    "method": method,
                    "para_index": pi,
                })
        return out
