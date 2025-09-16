# a0_logger.py
from __future__ import annotations
import os, csv
from typing import Dict, Iterable

class CSVLogger:
    def __init__(self, path: str, header: Iterable[str]):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._wr = csv.writer(self._fh)
        if write_header:
            self._wr.writerow(list(header))
            self._fh.flush()

    def log(self, row: Dict[str, object], fields: Iterable[str]):
        self._wr.writerow([row.get(k, "") for k in fields])
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass
