import os
import sqlite3
from typing import List, Dict


class Indexer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        try:
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS entries USING fts5(text, source, page, para_index, method)")
        except sqlite3.OperationalError as e:
            # FTS5 may not be available; fall back to a normal table
            self._conn.execute("CREATE TABLE IF NOT EXISTS entries (text TEXT, source TEXT, page INTEGER, para_index INTEGER, method TEXT)")
            self._conn.commit()

    def index(self, entries: List[Dict]):
        cur = self._conn.cursor()
        try:
            cur.executemany(
                "INSERT INTO entries(text, source, page, para_index, method) VALUES(?,?,?,?,?)",
                [(e.get("text", ""), e.get("source", ""), int(e.get("page", -1)), int(e.get("para_index", -1)), e.get("method", "")) for e in entries]
            )
        finally:
            self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
