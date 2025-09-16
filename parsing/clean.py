import re
import unicodedata
from typing import Dict


class Cleaner:
    def __init__(self):
        pass

    def fix_unicode(self, text: str) -> str:
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = unicodedata.normalize("NFC", text)
        return text

    def dehyphenate(self, text: str) -> str:
        # Join words split across line breaks or small spaces with hyphens
        # e.g., "environ-\nment" -> "environment", "environ- ment" -> "environment"
        text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[\t\r\f]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean(self, text: str) -> str:
        text = self.fix_unicode(text)
        text = self.dehyphenate(text)
        text = self.normalize_whitespace(text)
        return text
