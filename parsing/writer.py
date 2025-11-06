import csv
import json
import os
from typing import List, Dict, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(entries: List[Dict], out_path: str) -> int:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return len(entries)


def write_csv(entries: List[Dict], out_path: str) -> int:
    ensure_dir(os.path.dirname(out_path))
    if not entries:
        # create empty file with header
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "source", "page", "para_index", "method"]) 
            writer.writeheader()
        return 0
    fieldnames = sorted(set().union(*[e.keys() for e in entries]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            writer.writerow(e)
    return len(entries)
