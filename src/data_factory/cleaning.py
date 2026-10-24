"""Clean stage: turn validated text records into the typed case-level dataset.

Implements decisions D11-D16 and D24 of docs/decisions.md:
  - D24  a case closed after `as_of` counts as still open (independent downloads agree)
  - D15  open cases are kept, marked by is_open, with an empty days_to_close
  - D16  each case belongs to the month it was created in
  - D11  "Unspecified" style values become missing; out-of-city coordinates become missing
  - D12/D13 suspicious but possible values are flagged, never dropped
  - D14  duplicates: same address + type + descriptor + created second + closed second;
         one group = one problem; the row with the smallest unique_key represents it

Columns that keep changing in the live source after `as_of` (the source `status` and
`resolution_action_updated_date`) are not published, so the output does not depend on the
day it was downloaded. The status is only used to flag a contradiction with the dates.

Memory: the validated file is read in chunks. Duplicate detection needs to see every row,
so a first pass keeps only (group id, key) pairs; a second pass writes the output.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from data_factory.config import CleanConfig
from data_factory.exceptions import MissingInputError
from data_factory.tables import DATE_FORMAT, AtomicCsvWriter, read_csv_text
from data_factory.validation import VALIDATED_FILE

log = logging.getLogger(__name__)

CLEANED_FILE = "cleaned.csv"
STATS_FILE = "clean_stats.json"
CLOSED_STATUS = "CLOSED"
SECONDS_PER_DAY = 86_400
GROUP_ID_LENGTH = 16
MISSING_TOKEN = "<none>"   # stands for an empty closing time inside a duplicate key

CLEAN_COLUMNS = [
    "unique_key", "created_at", "closed_at", "is_open", "days_to_close", "created_month",
    "agency", "complaint_type", "descriptor", "incident_address", "incident_zip",
    "borough", "community_board", "latitude", "longitude", "open_data_channel_type",
    "flag_zero_duration", "flag_created_midnight", "flag_status_conflict", "flag_coords_invalid",
    "duplicate_group_id", "is_duplicate",
]
TEXT_COLUMNS = ["agency", "complaint_type", "descriptor", "incident_address", "incident_zip",
                "borough", "community_board", "open_data_channel_type"]


@dataclass(frozen=True)
class CleanStats:
    rows: int
    open_cases: int
    flag_zero_duration: int
    flag_created_midnight: int
    flag_status_conflict: int
    flag_coords_invalid: int
    duplicate_groups: int
    duplicate_rows: int          # rows marked is_duplicate (not counted again in summaries)


# ---------------------------------------------------------------- one chunk
def normalize_text(series: pd.Series, unspecified: tuple[str, ...] = ()) -> pd.Series:
    """Trim, collapse repeated spaces, and turn empty or 'unspecified' text into missing."""
    text = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    lowered = [value.lower() for value in unspecified]
    return text.mask((text == "") | text.str.lower().isin(lowered))


def _within(values: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    return values.between(bounds[0], bounds[1])


def clean_chunk(raw: pd.DataFrame, as_of: pd.Timestamp, settings: CleanConfig) -> pd.DataFrame:
    """Typed and derived columns for one chunk (duplicate columns are added later)."""
    out = pd.DataFrame(index=raw.index)
    out["unique_key"] = raw["unique_key"].astype("string")

    created = pd.to_datetime(raw["created_date"], format=DATE_FORMAT)
    closed = pd.to_datetime(raw["closed_date"], format=DATE_FORMAT, errors="coerce")
    closed = closed.where(closed <= as_of)                       # D24: after as_of = still open
    out["created_at"] = created
    out["closed_at"] = closed
    out["is_open"] = closed.isna()
    out["days_to_close"] = (closed - created).dt.total_seconds() / SECONDS_PER_DAY
    out["created_month"] = created.dt.strftime("%Y-%m")

    for column in TEXT_COLUMNS:
        out[column] = normalize_text(raw[column], settings.unspecified_values)
    out["incident_address"] = out["incident_address"].str.upper()

    latitude = pd.to_numeric(raw["latitude"], errors="coerce")
    longitude = pd.to_numeric(raw["longitude"], errors="coerce")
    given = raw["latitude"].notna() | raw["longitude"].notna()
    usable = _within(latitude, settings.latitude_range) & _within(longitude, settings.longitude_range)
    out["latitude"] = latitude.where(usable)
    out["longitude"] = longitude.where(usable)

    status = normalize_text(raw["status"]).str.upper()
    out["flag_zero_duration"] = (closed == created).fillna(False).astype(bool)
    out["flag_created_midnight"] = (created == created.dt.normalize())
    out["flag_status_conflict"] = (closed.notna() & status.notna() & (status != CLOSED_STATUS)).astype(bool)
    out["flag_coords_invalid"] = (given & ~usable).astype(bool)
    return out


def duplicate_group_ids(cleaned: pd.DataFrame) -> pd.Series:
    """D14 group id: hash of address, type, descriptor, created and closed second.

    Rows missing the address or the descriptor are never grouped (missing never matches).
    """
    parts = [
        cleaned["incident_address"],
        cleaned["complaint_type"],
        cleaned["descriptor"],
        cleaned["created_at"].dt.strftime(DATE_FORMAT),
        cleaned["closed_at"].dt.strftime(DATE_FORMAT).fillna(MISSING_TOKEN),
    ]
    groupable = parts[0].notna() & parts[2].notna()
    joined = parts[0].fillna("")
    for part in parts[1:]:
        joined = joined + "|" + part.fillna("")
    ids = joined.map(lambda text: hashlib.sha1(text.encode("utf-8")).hexdigest()[:GROUP_ID_LENGTH])
    return ids.where(groupable).astype("string")


# ---------------------------------------------------------------- whole dataset
def _chunks(path: Path, chunk_rows: int):
    yield from read_csv_text(path, chunksize=chunk_rows)


def _find_duplicates(path: Path, as_of: pd.Timestamp, settings: CleanConfig) -> tuple[set[str], int]:
    """Pass 1: keys of the rows that repeat a group's representative (its smallest key).

    Only (group id, key) pairs are kept in memory, not the records themselves.
    """
    pieces = []
    for raw in _chunks(path, settings.chunk_rows):
        cleaned = clean_chunk(raw, as_of, settings)
        pieces.append(pd.DataFrame({"group": duplicate_group_ids(cleaned),
                                    "key": cleaned["unique_key"]}))
    if not pieces:
        return set(), 0
    pairs = pd.concat(pieces, ignore_index=True).dropna(subset=["group"])
    pairs["number"] = pairs["key"].astype("int64")       # keys are digits (contract)
    grouped = pairs[pairs.groupby("group")["key"].transform("size") > 1]
    representative = grouped.groupby("group")["number"].transform("min")
    extra = grouped.loc[grouped["number"] != representative, "key"]
    return set(extra.tolist()), int(grouped["group"].nunique())


def clean_dataset(work_dir: Path, as_of: pd.Timestamp, settings: CleanConfig) -> CleanStats:
    source = work_dir / VALIDATED_FILE
    if not source.exists():
        raise MissingInputError(f"{source} not found: run the validate stage first")

    duplicate_keys, duplicate_groups = _find_duplicates(source, as_of, settings)

    counts = dict.fromkeys(["rows", "open_cases", "flag_zero_duration", "flag_created_midnight",
                            "flag_status_conflict", "flag_coords_invalid", "duplicate_rows"], 0)
    with AtomicCsvWriter(work_dir / CLEANED_FILE, CLEAN_COLUMNS) as writer:
        for raw in _chunks(source, settings.chunk_rows):          # pass 2
            cleaned = clean_chunk(raw, as_of, settings)
            cleaned["duplicate_group_id"] = duplicate_group_ids(cleaned)
            cleaned["is_duplicate"] = cleaned["unique_key"].isin(duplicate_keys).astype(bool)
            writer.append(cleaned)
            counts["rows"] += len(cleaned)
            counts["open_cases"] += int(cleaned["is_open"].sum())
            counts["duplicate_rows"] += int(cleaned["is_duplicate"].sum())
            for flag in ("flag_zero_duration", "flag_created_midnight",
                         "flag_status_conflict", "flag_coords_invalid"):
                counts[flag] += int(cleaned[flag].sum())
        writer.commit()

    stats = CleanStats(duplicate_groups=duplicate_groups, **counts)
    (work_dir / STATS_FILE).write_text(json.dumps(asdict(stats), indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8", newline="\n")
    log.info("clean: %d rows, %d open, %d duplicate rows in %d groups",
             stats.rows, stats.open_cases, stats.duplicate_rows, stats.duplicate_groups)
    return stats
