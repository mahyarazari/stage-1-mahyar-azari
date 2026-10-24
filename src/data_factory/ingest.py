"""Download a source into a raw snapshot: page by page, resumable and idempotent.

  - Resumable: an interrupted run continues after the last complete page.
  - Idempotent: a complete snapshot is never downloaded or changed again.
  - Bounded memory: one page in memory at a time.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from data_factory.contract import Scope
from data_factory.exceptions import InvalidResponseError
from data_factory.raw_store import RawPageStore
from data_factory.sources import Source

log = logging.getLogger(__name__)

SNAPSHOT_ID_LENGTH = 12


@dataclass(frozen=True)
class FetchResult:
    pages_written: int
    records_written: int
    already_complete: bool


def snapshot_id(records_url: str, columns: list[str], scope: Scope) -> str:
    """Name of the snapshot folder: changes whenever what is downloaded changes.

    Changing the window, the scope or the columns therefore starts a new snapshot instead of
    mixing old and new pages in one folder.
    """
    identity = {
        "records_url": records_url,
        "columns": sorted(columns),
        "agencies": sorted(scope.agencies),
        "complaint_types": sorted(scope.complaint_types),
        "window_start": scope.window_start.isoformat(),
        "window_end": scope.window_end.isoformat(),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return f"snapshot-{digest[:SNAPSHOT_ID_LENGTH]}"


def _check_page_order(page: list[dict], key_field: str, after_key: str | None) -> None:
    """Keyset pagination is only safe if keys strictly increase; refuse anything else."""
    keys = [record.get(key_field) for record in page]
    if any(not isinstance(key, str) for key in keys):
        raise InvalidResponseError(f"a record in the page has no text '{key_field}'")
    previous = after_key
    for key in keys:
        if previous is not None and key <= previous:
            raise InvalidResponseError(
                f"keys are not strictly increasing ({previous!r} then {key!r}); "
                "resuming from this page could skip or repeat records")
        previous = key


def fetch_snapshot(source: Source, store: RawPageStore, expected_columns: dict[str, str],
                   page_size: int, now: datetime | None = None,
                   extra_info: dict | None = None) -> FetchResult:
    source.check_schema(expected_columns)

    if store.is_complete():
        log.info("snapshot %s is complete: nothing to download", store.directory.name)
        return FetchResult(pages_written=0, records_written=0, already_complete=True)

    removed = store.remove_temporary_files()
    if removed:
        log.warning("removed %d unfinished page file(s) from an interrupted run", removed)

    after_key = store.last_key()
    page_number = store.next_page_number()
    if after_key is not None:
        log.info("resuming after key %s (page %d)", after_key, page_number)

    pages_written = records_written = 0
    while True:
        page = source.fetch_page(after_key, page_size)
        if not page:
            break
        _check_page_order(page, source.key_field, after_key)
        store.write_page(page_number, page)
        pages_written += 1
        records_written += len(page)
        log.info("page %d: %d records (last key %s)", page_number, len(page),
                 page[-1][source.key_field])
        after_key = page[-1][source.key_field]
        page_number += 1
        if len(page) < page_size:   # a short page is the last one: no extra empty request
            break

    # UTC, not the local clock: the result must not depend on the machine's time zone.
    # (Source dates are New York local time, which is always behind UTC, so using UTC as the
    # "now" bound is a few hours generous but never rejects a real closing date.)
    finished_at = (now or datetime.now(timezone.utc).replace(tzinfo=None)).replace(microsecond=0)
    store.mark_complete({
        "fetched_at_utc": finished_at.isoformat(),
        "pages": len(store.page_files()),
        "records": store.count_records(),
        **(extra_info or {}),
    })
    return FetchResult(pages_written, records_written, already_complete=False)
