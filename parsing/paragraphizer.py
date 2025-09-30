from typing import List, Dict
import re


class Paragraphizer:
    """Segments page texts into paragraphs per page (self-contained).

    Rules:
    - Break paragraphs on blank lines, all-caps headers, and numbered sections (e.g., "1. ")
    - Keep paragraphs within length and sentence count bounds
    - Long paragraphs are split on sentence boundaries instead of being dropped
    """

    def __init__(self,
                 min_chars: int = 50,
                 max_chars: int = 2000,
                 min_sentences: int = 2,
                 max_sentences: int = 15):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def _is_section_break(self, line: str) -> bool:
        """Detect section breaks: blank line, all-caps header, or numbered section."""
        if not line:
            return True
        if re.match(r'^[A-Z\s]{10,}$', line):
            return True
        if re.match(r'^\d+\.\s', line):
            return True
        return False

    def _count_sentences(self, text: str) -> int:
        return len(re.findall(r'[.!?]+', text))

    def _paragraph_dict(self, text: str) -> Dict[str, str]:
        text = text.strip()
        return {
            'text': text,
            'char_count': len(text),
            'sentence_count': self._count_sentences(text),
        }

    def _split_long_text(self, text: str) -> List[Dict[str, str]]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[str] = []
        current: List[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            tentative = " ".join(current + [sentence]).strip()
            if current and (len(tentative) > self.max_chars or self._count_sentences(tentative) > self.max_sentences):
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current = [sentence]
            else:
                current.append(sentence)
        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

        if not chunks:
            return [self._paragraph_dict(text)]

        normalized_chunks = [c.strip() for c in chunks if c.strip()]
        if not normalized_chunks:
            return [self._paragraph_dict(text)]
        if len(normalized_chunks) == 1 and normalized_chunks[0] == text.strip():
            return [self._paragraph_dict(text)]

        paragraphs: List[Dict[str, str]] = []
        carryover = ""
        for chunk in normalized_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            if carryover:
                chunk = f"{carryover} {chunk}".strip()
                carryover = ""

            char_count = len(chunk)
            sentence_count = self._count_sentences(chunk)

            if char_count < self.min_chars and sentence_count < self.min_sentences:
                carryover = chunk
                continue

            if char_count > self.max_chars or sentence_count > self.max_sentences:
                paragraphs.extend(self._split_long_text(chunk))
            else:
                paragraphs.append(self._paragraph_dict(chunk))

        if carryover:
            if paragraphs:
                combined = f"{paragraphs[-1]['text']} {carryover}".strip()
                if len(combined) > self.max_chars or self._count_sentences(combined) > self.max_sentences:
                    last = paragraphs.pop()
                    paragraphs.extend(self._split_long_text(f"{last['text']} {carryover}".strip()))
                else:
                    paragraphs[-1] = self._paragraph_dict(combined)
            else:
                paragraphs.append(self._paragraph_dict(carryover))

        return paragraphs or [self._paragraph_dict(text)]

    def _create_paragraph(self, text: str) -> List[Dict[str, str]]:
        text = text.strip()
        if not text:
            return []

        char_count = len(text)
        sentence_count = self._count_sentences(text)

        if char_count <= self.max_chars and sentence_count <= self.max_sentences:
            if char_count >= self.min_chars and sentence_count >= self.min_sentences:
                return [self._paragraph_dict(text)]
            if char_count >= self.min_chars or sentence_count >= self.min_sentences:
                return [self._paragraph_dict(text)]
            return []

        return self._split_long_text(text)

    def segment_text(self, text: str) -> List[Dict[str, str]]:
        paragraphs: List[Dict[str, str]] = []
        current_text = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if self._is_section_break(line):
                if current_text:
                    paras = self._create_paragraph(current_text)
                    if paras:
                        paragraphs.extend(paras)
                    current_text = ""
                continue
            current_text += line + " "
        if current_text:
            paras = self._create_paragraph(current_text)
            if paras:
                paragraphs.extend(paras)

        return paragraphs

    def paragraphs_from_pages(self, pages: List[Dict]) -> List[Dict]:
        """Input: list of {page, text, method, ...}
        Output: list of paragraphs with metadata: {text, page, method, para_index}
        """
        out: List[Dict] = []
        for page_entry in pages:
            page_num = page_entry["page"]
            method = page_entry.get("method", "text")
            text = page_entry.get("text", "")
            paras = self.segment_text(text)
            if not paras and text.strip():
                paras = [self._paragraph_dict(text)]
            for pi, p in enumerate(paras):
                out.append({
                    "text": p["text"],
                    "page": page_num,
                    "method": method,
                    "para_index": pi,
                })
        return out
