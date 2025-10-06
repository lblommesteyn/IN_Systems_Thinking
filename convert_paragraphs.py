#!/usr/bin/env python3
"""
convert_paragraphs.py
--------------------

Usage
-----
$ python convert_paragraphs.py

Converts all JSON files in the Unilever_Nestle_testing directory to CSV format.
Handles JSON files with either a top-level 'paragraphs' key or direct lists of objects.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file, handling multiple formats.
    
    Supports:
    - Direct lists of objects
    - JSON with a 'paragraphs' key
    - JSON with a 'cross_scale_paragraphs' key (Claude 3.5 format)
    - JSON with a 'cross_scale_connections' key (Claude 3.7 format)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle direct list format
    if isinstance(data, list):
        return data
    # Handle 'paragraphs' key format
    elif "paragraphs" in data and isinstance(data["paragraphs"], list):
        return data["paragraphs"]
    # Handle 'cross_scale_paragraphs' key format (Claude 3.5 files)
    elif "cross_scale_paragraphs" in data and isinstance(data["cross_scale_paragraphs"], list):
        return data["cross_scale_paragraphs"]
    # Handle 'cross_scale_connections' key format (Claude 3.7 files)
    elif "cross_scale_connections" in data and isinstance(data["cross_scale_connections"], list):
        return data["cross_scale_connections"]
    else:
        raise ValueError("JSON must either be a list or contain a list under one of the keys: 'paragraphs', 'cross_scale_paragraphs', or 'cross_scale_connections'.")


def collect_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """Grab a superset of keys and put the useful ones first."""
    preferred = [
        "paragraph_text",
        "entities",
        "relationship_type",
        "cross_scale_connection",
        "feedback_loop_present",
        "loop_nature",
        "systemic_factor",
        "strength_rating",
        "explicitness",
        "merged_context",
        "pdf_source",
        "confidence",
        "citation",
    ]
    seen = {key for row in rows for key in row}
    ordered = preferred + [k for k in sorted(seen) if k not in preferred]
    return ordered


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    """Dump rows to CSV, stringifying any lists/dicts so Excel/Sheets behave."""
    fieldnames = collect_fieldnames(rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean_row = {
                k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                for k, v in row.items()
            }
            writer.writerow(clean_row)
    print(f"✅  Saved {len(rows)} rows to '{out_path}'")


def process_json_file(json_path: str) -> None:
    """Process a single JSON file and convert it to CSV."""
    # Create output path with same name but .csv extension
    csv_path = json_path.replace('.json', '.csv')
    
    try:
        rows = load_json(json_path)
        write_csv(rows, csv_path)
    except Exception as e:
        print(f"❌ Error processing {json_path}: {e}")


def main() -> None:
    # Path to the Unilever_Nestle_testing directory
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Unilever_Nestle_testing')
    
    if not os.path.exists(dir_path):
        print(f"❌ Directory not found: {dir_path}")
        sys.exit(1)
    
    # Find all JSON files in the directory
    json_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                 if f.endswith('.json') and os.path.isfile(os.path.join(dir_path, f))]
    
    if not json_files:
        print(f"❌ No JSON files found in {dir_path}")
        sys.exit(1)
    
    print(f"🔍 Found {len(json_files)} JSON files to process")
    
    # Process each JSON file
    for json_file in json_files:
        process_json_file(json_file)
    
    print(f"✅ Conversion complete! All JSON files have been converted to CSV format.")


if __name__ == "__main__":
    main()
