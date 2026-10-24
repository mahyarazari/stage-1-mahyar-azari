"""Write and read tables (CSV) the same way in every stage, so outputs are deterministic.

Why CSV: it is readable by people and by PostgreSQL `COPY` (stage 2) and needs no library
outside the allowed tools (Parquet would need pyarrow).

Determinism rules (D20, D24): UTF-8, LF line endings, fixed date format, fixed float
format, no index column, and a file only appears under its final name once complete.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

ENCODING = "utf-8"
LINE_END = "\n"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
FLOAT_FORMAT = "%.6g"
TEMP_SUFFIX = ".tmp"


class AtomicCsvWriter:
    """Append DataFrames to one CSV; the file gets its final name only on `commit()`.

    Used as a context manager: if anything fails before `commit()`, the temporary file is
    removed and no half-written output is left behind.
    """

    def __init__(self, path: Path, columns: list[str]):
        self.path = Path(path)
        self.columns = list(columns)
        self.temporary = self.path.with_name(self.path.name + TEMP_SUFFIX)
        self.rows = 0
        self._handle = None

    def __enter__(self) -> "AtomicCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary.open("w", encoding=ENCODING, newline="")
        pd.DataFrame(columns=self.columns).to_csv(self._handle, index=False, lineterminator=LINE_END)
        return self

    def append(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        frame.reindex(columns=self.columns).to_csv(
            self._handle, index=False, header=False, lineterminator=LINE_END,
            date_format=DATE_FORMAT, float_format=FLOAT_FORMAT)
        self.rows += len(frame)

    def commit(self) -> None:
        self._close()
        os.replace(self.temporary, self.path)

    def discard(self) -> None:
        self._close()
        if self.temporary.exists():
            self.temporary.unlink()

    def _close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None or self.temporary.exists():
            self.discard()   # never committed: leave nothing behind


def write_csv(path: Path, frame: pd.DataFrame) -> int:
    """Write one complete table atomically. Returns the number of rows written."""
    with AtomicCsvWriter(path, list(frame.columns)) as writer:
        writer.append(frame)
        writer.commit()
    return writer.rows


def read_csv_text(path: Path, chunksize: int | None = None) -> pd.DataFrame | Iterable[pd.DataFrame]:
    """Read a table with every column as text; empty cells become missing values."""
    return pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[""],
                       encoding=ENCODING, chunksize=chunksize)
