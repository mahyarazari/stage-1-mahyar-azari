"""Raw snapshot on disk: pages exactly as the source sent them, one JSON-lines file per page.

Design (D22, D23 in docs/decisions.md):
  - Each page is written to a temporary name and renamed only when complete (an atomic
    operation), so a half-written page can never be mistaken for a complete one.
  - The completed page files ARE the checkpoint: the resume point is the last key of the
    last complete page. There is no separate state file that could disagree with them.
  - A COMPLETE marker is written after the last page. Once it exists the snapshot is never
    changed again: later runs read it instead of downloading (D24, determinism).
  - Only one page is ever held in memory, when writing or when reading back.

The store also works as a *file source*: `read_pages()` replays the snapshot page by page,
so every later stage reads raw data without knowing it once came over HTTP.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

from data_factory.exceptions import DataValidationError

PAGE_PATTERN = re.compile(r"page-(\d{5})\.jsonl")
TEMP_SUFFIX = ".tmp"
COMPLETE_MARKER = "COMPLETE.json"
ENCODING = "utf-8"
NEWLINE = "\n"   # explicit: the same bytes on Windows and Linux (D20)


def _write_atomically(path: Path, lines: Iterator[str]) -> None:
    temporary = path.with_name(path.name + TEMP_SUFFIX)
    with temporary.open("w", encoding=ENCODING, newline=NEWLINE) as handle:
        for line in lines:
            handle.write(line + NEWLINE)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)   # atomic on Windows and POSIX


def _to_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


class RawPageStore:
    def __init__(self, directory: Path, key_field: str = "unique_key"):
        self.directory = Path(directory)
        self.key_field = key_field

    # ------------------------------------------------------------ reading state
    def page_files(self) -> list[Path]:
        """Complete pages, in order. Temporary (unfinished) files are ignored."""
        if not self.directory.exists():
            return []
        pages = [p for p in self.directory.iterdir() if PAGE_PATTERN.fullmatch(p.name)]
        return sorted(pages)

    def is_complete(self) -> bool:
        return (self.directory / COMPLETE_MARKER).exists()

    def completion_info(self) -> dict[str, Any]:
        with (self.directory / COMPLETE_MARKER).open(encoding=ENCODING) as handle:
            return json.load(handle)

    def last_key(self) -> str | None:
        """Resume point: key of the last record of the last complete page (None = start)."""
        pages = self.page_files()
        if not pages:
            return None
        last_line = ""
        with pages[-1].open(encoding=ENCODING) as handle:
            for line in handle:      # pages are small enough to scan; nothing is kept
                if line.strip():
                    last_line = line
        if not last_line:
            raise DataValidationError(f"raw page {pages[-1].name} is empty")
        return json.loads(last_line)[self.key_field]

    def next_page_number(self) -> int:
        pages = self.page_files()
        return 1 if not pages else int(PAGE_PATTERN.fullmatch(pages[-1].name).group(1)) + 1

    # ------------------------------------------------------------ writing
    def write_page(self, number: int, records: list[dict[str, Any]]) -> Path:
        if self.is_complete():
            raise DataValidationError(f"snapshot {self.directory} is complete and must not change")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"page-{number:05d}.jsonl"
        _write_atomically(path, (_to_line(record) for record in records))
        return path

    def mark_complete(self, info: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        text = json.dumps(info, ensure_ascii=False, sort_keys=True, indent=2)
        _write_atomically(self.directory / COMPLETE_MARKER, iter([text]))

    def remove_temporary_files(self) -> int:
        """Leftovers of an interrupted write; safe to delete because they were never complete."""
        if not self.directory.exists():
            return 0
        leftovers = [p for p in self.directory.iterdir() if p.name.endswith(TEMP_SUFFIX)]
        for path in leftovers:
            path.unlink()
        return len(leftovers)

    def count_records(self) -> int:
        total = 0
        for path in self.page_files():
            with path.open(encoding=ENCODING) as handle:
                total += sum(1 for line in handle if line.strip())
        return total

    # ------------------------------------------------------------ file source
    def read_pages(self) -> Iterator[list[dict[str, Any]]]:
        """Replay the snapshot one page at a time."""
        for path in self.page_files():
            with path.open(encoding=ENCODING) as handle:
                yield [json.loads(line) for line in handle if line.strip()]
