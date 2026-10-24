"""Data contract for the case-level dataset.

The contract answers two questions:
  1. What does the source have to look like? (SOURCE_COLUMNS, used to detect schema changes)
  2. When is a single record broken? (validate(), which never drops a record silently:
     every rejected record goes to quarantine with the reasons it was rejected.)

The contract knows nothing about HTTP, files or databases. It receives a DataFrame of raw
text values and returns two DataFrames, so any source and any sink can reuse it.

Policy summary (the "why" lives in report.md):
  - A record is quarantined only when it is broken. A healthy record that merely lacks
    an optional value (zip, coordinates, closed_date) passes; later stages flag it.
  - Membership of a zip in a reference table is NOT checked here: references can be
    outdated (see zip 11249), and quarantining on an outdated reference would throw
    away healthy records. Only the zip *format* is part of the contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
REASON_SEPARATOR = "|"
REASON_COLUMN = "reason_codes"


class ReasonCode(str, Enum):
    """Why a record was quarantined. The order here is the order reasons are written in."""

    MISSING_REQUIRED = "MISSING_REQUIRED"
    INVALID_KEY = "INVALID_KEY"
    UNPARSEABLE_DATE = "UNPARSEABLE_DATE"
    OUT_OF_WINDOW = "OUT_OF_WINDOW"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    CLOSED_BEFORE_CREATED = "CLOSED_BEFORE_CREATED"
    CLOSED_IN_FUTURE = "CLOSED_IN_FUTURE"
    ZIP_MALFORMED = "ZIP_MALFORMED"
    BOROUGH_INVALID = "BOROUGH_INVALID"


# Columns requested from the source, with their type in the source metadata.
# A column that disappears or changes type here means the source changed shape.
SOURCE_COLUMNS: dict[str, str] = {
    "unique_key": "text",
    "created_date": "calendar_date",
    "closed_date": "calendar_date",
    "status": "text",
    "agency": "text",
    "complaint_type": "text",
    "descriptor": "text",
    "incident_zip": "text",
    "incident_address": "text",
    "borough": "text",
    "community_board": "text",
    "latitude": "number",
    "longitude": "number",
    "open_data_channel_type": "text",
    "resolution_action_updated_date": "calendar_date",
}

REQUIRED_COLUMNS = ("unique_key", "created_date", "agency", "complaint_type")
DATE_COLUMNS = tuple(name for name, kind in SOURCE_COLUMNS.items() if kind == "calendar_date")

BOROUGHS = ("BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND")
BOROUGH_UNSPECIFIED = "Unspecified"


@dataclass(frozen=True)
class Scope:
    """Which records belong to this run. Built from configuration, never hard-coded."""

    agencies: tuple[str, ...]
    complaint_types: tuple[str, ...]
    window_start: pd.Timestamp  # inclusive
    window_end: pd.Timestamp    # exclusive


@dataclass(frozen=True)
class ValidationResult:
    valid: pd.DataFrame        # normalized text values, date columns parsed
    quarantined: pd.DataFrame  # original raw values plus the reason_codes column


def _blank_to_na(series: pd.Series) -> pd.Series:
    """Text with surrounding spaces removed; empty text becomes missing."""
    text = series.astype("string").str.strip()
    return text.mask(text == "")


def normalize_zip(series: pd.Series, placeholders: tuple[str, ...]) -> pd.Series:
    """Turn known placeholder values (e.g. 'N/A', '00000') into missing and ZIP+4 into 5 digits."""
    text = _blank_to_na(series)
    upper_placeholders = [value.upper() for value in placeholders]
    text = text.mask(text.str.upper().isin(upper_placeholders))
    return text.str.replace(r"^(\d{5})-\d{4}$", r"\1", regex=True)


def _no(mask: pd.Series) -> pd.Series:
    """Unknown (missing) answers to a check count as 'not violated'."""
    return mask.fillna(False).astype(bool)


def validate(
    raw: pd.DataFrame,
    scope: Scope,
    fetched_at: pd.Timestamp,
    zip_placeholders: tuple[str, ...],
) -> ValidationResult:
    """Split raw records into valid and quarantined ones.

    `fetched_at` is when the data was downloaded: a case cannot have been closed after it.
    (A case closed after the configured `as_of` cut-off but before `fetched_at` is real and
    valid; the cleaning stage treats it as still open, so independent downloads agree.)

    `raw` holds text exactly as the source sent it. Columns the source omitted
    (the API leaves out empty fields) are treated as missing values, not as errors;
    detecting a column removed from the source is the job of the schema check.
    """
    original = raw.reindex(columns=list(SOURCE_COLUMNS))
    text = original.apply(_blank_to_na)
    text["incident_zip"] = normalize_zip(text["incident_zip"], zip_placeholders)

    parsed = {
        column: pd.to_datetime(text[column], format=DATE_FORMAT, errors="coerce")
        for column in DATE_COLUMNS
    }
    created, closed = parsed["created_date"], parsed["closed_date"]

    unparseable_date = pd.Series(False, index=text.index)
    for column in DATE_COLUMNS:
        unparseable_date |= text[column].notna() & parsed[column].isna()

    checks = {
        ReasonCode.MISSING_REQUIRED: text[list(REQUIRED_COLUMNS)].isna().any(axis=1),
        ReasonCode.INVALID_KEY: text["unique_key"].notna()
        & ~_no(text["unique_key"].str.fullmatch(r"\d+")),
        ReasonCode.UNPARSEABLE_DATE: unparseable_date,
        ReasonCode.OUT_OF_WINDOW: created.notna()
        & ((created < scope.window_start) | (created >= scope.window_end)),
        ReasonCode.OUT_OF_SCOPE: (text["agency"].notna() & ~text["agency"].isin(scope.agencies))
        | (text["complaint_type"].notna() & ~text["complaint_type"].isin(scope.complaint_types)),
        ReasonCode.CLOSED_BEFORE_CREATED: closed < created,
        ReasonCode.CLOSED_IN_FUTURE: closed > fetched_at,
        ReasonCode.ZIP_MALFORMED: text["incident_zip"].notna()
        & ~_no(text["incident_zip"].str.fullmatch(r"\d{5}")),
        ReasonCode.BOROUGH_INVALID: text["borough"].notna()
        & ~text["borough"].isin([*BOROUGHS, BOROUGH_UNSPECIFIED]),
    }
    violated = pd.DataFrame({code: _no(mask) for code, mask in checks.items()}, index=text.index)
    broken = violated.any(axis=1)

    # Reasons are written in ReasonCode order, so the text is deterministic.
    reasons = pd.Series("", index=text.index, dtype="string")
    for code in ReasonCode:
        reasons = reasons.where(~violated[code], reasons + code.value + REASON_SEPARATOR)
    reasons = reasons.str.rstrip(REASON_SEPARATOR)

    quarantined = original.loc[broken].copy()
    quarantined[REASON_COLUMN] = reasons.loc[broken]

    valid = text.loc[~broken].copy()
    for column in DATE_COLUMNS:
        valid[column] = parsed[column].loc[~broken]

    return ValidationResult(valid=valid, quarantined=quarantined)


def reason_counts(quarantined: pd.DataFrame) -> dict[str, int]:
    """Number of quarantined records per reason. A record with two reasons counts in both."""
    counts = {code.value: 0 for code in ReasonCode}
    if quarantined.empty:
        return counts
    exploded = quarantined[REASON_COLUMN].str.split(REASON_SEPARATOR).explode()
    for code, number in exploded.value_counts().items():
        counts[code] = int(number)
    return counts
