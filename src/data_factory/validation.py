"""Validate stage: run the contract over the raw snapshot, one page at a time.

Outputs (all written atomically, so a failed run leaves no half-written file):
  - <work>/<snapshot>/validated.csv        records that passed the contract
  - <quarantine>/<snapshot>/quarantine.csv rejected records, raw values + reason_codes
  - <work>/<snapshot>/validate_stats.json  counts per reason, used by the manifest

If the quarantined share is above the configured threshold, the quarantine file is still
written (it is the evidence) but validated.csv is not: the run stops with exit code 31.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import pandas as pd

from data_factory.contract import (
    REASON_COLUMN,
    SOURCE_COLUMNS,
    Scope,
    reason_counts,
    validate,
)
from data_factory.exceptions import MissingInputError, QuarantineThresholdError
from data_factory.raw_store import RawPageStore
from data_factory.tables import AtomicCsvWriter

log = logging.getLogger(__name__)

VALIDATED_FILE = "validated.csv"
QUARANTINE_FILE = "quarantine.csv"
STATS_FILE = "validate_stats.json"


@dataclass(frozen=True)
class ValidationStats:
    raw_records: int
    valid_records: int
    quarantined_records: int
    quarantine_ratio: float
    max_quarantine_ratio: float
    reasons: dict[str, int]


def validate_snapshot(store: RawPageStore, work_dir, quarantine_dir, scope: Scope,
                      zip_placeholders: tuple[str, ...], max_ratio: float) -> ValidationStats:
    if not store.is_complete():
        raise MissingInputError(f"raw snapshot {store.directory} is not complete: run the fetch stage first")
    fetched_at = pd.Timestamp(store.completion_info()["fetched_at_utc"])

    columns = list(SOURCE_COLUMNS)
    reasons = reason_counts(pd.DataFrame(columns=[REASON_COLUMN]))   # every reason at 0
    raw_records = 0

    with AtomicCsvWriter(work_dir / VALIDATED_FILE, columns) as valid_out, \
         AtomicCsvWriter(quarantine_dir / QUARANTINE_FILE, columns + [REASON_COLUMN]) as bad_out:
        for page in store.read_pages():            # one page in memory at a time
            result = validate(pd.DataFrame(page), scope, fetched_at, zip_placeholders)
            raw_records += len(page)
            valid_out.append(result.valid)
            bad_out.append(result.quarantined)
            for reason, count in reason_counts(result.quarantined).items():
                reasons[reason] += count

        ratio = bad_out.rows / raw_records if raw_records else 0.0
        stats = ValidationStats(raw_records=raw_records, valid_records=valid_out.rows,
                                quarantined_records=bad_out.rows,
                                quarantine_ratio=round(ratio, 8),
                                max_quarantine_ratio=max_ratio, reasons=reasons)
        bad_out.commit()                            # the evidence is kept in every case
        log.info("validate: %d raw, %d valid, %d quarantined (%.4f%%)",
                 raw_records, valid_out.rows, bad_out.rows, 100 * ratio)
        for reason, count in reasons.items():
            if count:
                log.info("  %-24s %d", reason, count)

        if ratio > max_ratio:
            raise QuarantineThresholdError(
                f"{100 * ratio:.4f}% of records were quarantined, above the limit of "
                f"{100 * max_ratio:.4f}%; see {quarantine_dir / QUARANTINE_FILE}")
        valid_out.commit()

    _write_stats(work_dir / STATS_FILE, stats)
    return stats


def _write_stats(path, stats: ValidationStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(stats), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def read_stats(work_dir) -> dict:
    path = work_dir / STATS_FILE
    if not path.exists():
        raise MissingInputError(f"{path} not found: run the validate stage first")
    return json.loads(path.read_text(encoding="utf-8"))
