import argparse
import os
from datetime import datetime
from typing import List, Dict

from parsing.loader import Loader
from parsing.ocr import OCR, OCRConfig
from parsing.extractor import Extractor, ExtractionConfig
from parsing.paragraphizer import Paragraphizer
from parsing.clean import Cleaner
from parsing.dedup import Deduplicator
from parsing.writer import write_jsonl, write_csv
from parsing.indexer import Indexer


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PDF -> paragraphs pipeline with OCR fallback and optional FTS5 index")
    p.add_argument("input", help="Path to a PDF file")
    p.add_argument("--out-dir", default=os.path.join("data", "processed"), help="Output directory")
    p.add_argument("--basename", default=None, help="Base name for output files (default: input stem)")
    p.add_argument("--jsonl", action="store_true", help="Write JSONL output")
    p.add_argument("--csv", action="store_true", help="Write CSV output")
    p.add_argument("--index", action="store_true", help="Insert into SQLite FTS5 index")
    p.add_argument("--db-path", default=os.path.join("data", "processed", "fts5_index.db"), help="Path to SQLite database for FTS5 index")
    p.add_argument("--lang", default="eng", help="Tesseract language, e.g., 'eng' or 'eng+deu'")
    p.add_argument("--min-chars", type=int, default=50, help="Minimum chars to consider a page as text (else OCR)")
    p.add_argument("--min-words", type=int, default=10, help="Minimum words to consider a page as text (else OCR)")
    p.add_argument("--dedup-threshold", type=float, default=0.8, help="Near-duplicate cosine similarity threshold")
    p.add_argument("--tesseract-cmd", default=None, help="Full path to tesseract executable (Windows)")
    p.add_argument("--dpi", type=int, default=300, help="DPI for page rendering before OCR (higher = slower, better OCR)")
    p.add_argument("--no-dedup", action="store_true", help="Disable near-duplicate filtering")
    return p


def main():
    args = build_arg_parser().parse_args()

    src_path = args.input
    if not os.path.exists(src_path):
        raise SystemExit(f"Input not found: {src_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    base = args.basename or os.path.splitext(os.path.basename(src_path))[0]

    # Initialize components
    loader = Loader()
    ocr = OCR(OCRConfig(lang=args.lang, dpi=args.dpi, tesseract_cmd=args.tesseract_cmd))
    extractor = Extractor(ExtractionConfig(min_chars=args.min_chars, min_words=args.min_words))
    para = Paragraphizer()
    cleaner = Cleaner()
    dedup = Deduplicator(threshold=args.dedup_threshold)

    # Load and extract
    doc = loader.open_pdf(src_path)
    pages = extractor.extract_pages(doc, ocr=ocr)

    # Paragraphize
    paragraphs = para.paragraphs_from_pages(pages)

    # Clean
    for p in paragraphs:
        p["text"] = cleaner.clean(p["text"])
        p["source"] = src_path

    # Deduplicate (optional - enabled by default with threshold)
    if not args.no_dedup:
        paragraphs = dedup.deduplicate(paragraphs)

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(args.out_dir, f"{base}.jsonl")
    csv_path = os.path.join(args.out_dir, f"{base}.csv")

    wrote_any = False
    if args.jsonl or (not args.csv and not args.jsonl):
        # default to JSONL if no format flags provided
        n_jsonl = write_jsonl(paragraphs, jsonl_path)
        wrote_any = True
    else:
        n_jsonl = 0

    if args.csv:
        n_csv = write_csv(paragraphs, csv_path)
        wrote_any = True
    else:
        n_csv = 0

    # Index (optional)
    n_indexed = 0
    if args.index:
        indexer = Indexer(args.db_path)
        indexer.index(paragraphs)
        indexer.close()
        n_indexed = len(paragraphs)

    # Report
    outputs = []
    if n_jsonl:
        outputs.append(jsonl_path)
    if n_csv:
        outputs.append(csv_path)
    out_paths = ", ".join(outputs) if outputs else "(no files written)"

    print("done")
    print(f"Input: {src_path}")
    print(f"Output: {out_paths}")
    print(f"Counts -> pages: {len(pages)}, paragraphs: {len(paragraphs)}, indexed: {n_indexed}")


if __name__ == "__main__":
    main()
