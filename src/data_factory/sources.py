"""Sources: where records come from.

A source must be able to (1) check that it still has the shape the contract expects and
(2) hand over records page by page, in increasing key order, starting after a given key.
Anything with those two methods can feed the factory: the Socrata API today, a file or
another API later, without touching validation, quarantine or the manifest.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

import pandas as pd

from data_factory.contract import Scope
from data_factory.exceptions import InvalidResponseError, SourceSchemaChangedError
from data_factory.http_client import HttpClient

log = logging.getLogger(__name__)

SOCRATA_TIMESTAMP = "%Y-%m-%dT%H:%M:%S"


class Source(Protocol):
    key_field: str

    def check_schema(self, expected: dict[str, str]) -> None: ...

    def fetch_page(self, after_key: str | None, limit: int) -> list[dict[str, Any]]: ...


def _quote(value: str) -> str:
    """A SoQL text literal; a single quote inside the value is doubled."""
    return "'" + value.replace("'", "''") + "'"


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} in({', '.join(_quote(v) for v in values)})"


class SocrataSource:
    """A Socrata (SODA 2.x) dataset read with keyset pagination on `key_field` (D22)."""

    def __init__(self, http: HttpClient, records_url: str, metadata_url: str,
                 scope: Scope, columns: list[str], key_field: str = "unique_key",
                 date_field: str = "created_date"):
        self.http = http
        self.records_url = records_url
        self.metadata_url = metadata_url
        self.scope = scope
        self.columns = columns
        self.key_field = key_field
        self.date_field = date_field

    # ------------------------------------------------------------ schema
    def check_schema(self, expected: dict[str, str]) -> None:
        """Stop if a needed column is gone or changed type. Extra columns are fine."""
        metadata = self.http.get_json(self.metadata_url)
        columns = metadata.get("columns") if isinstance(metadata, dict) else None
        if not isinstance(columns, list):
            raise InvalidResponseError(f"metadata from {self.metadata_url} has no column list")
        actual = {c.get("fieldName"): c.get("dataTypeName") for c in columns if isinstance(c, dict)}

        missing = sorted(name for name in expected if name not in actual)
        changed = sorted(f"{name}: {expected[name]} -> {actual[name]}"
                         for name in expected if name in actual and actual[name] != expected[name])
        if missing or changed:
            raise SourceSchemaChangedError(
                f"source schema changed; missing columns: {missing or 'none'}; "
                f"type changes: {changed or 'none'}")
        log.info("source schema ok: %d expected columns present with expected types", len(expected))

    # ------------------------------------------------------------ records
    def where_clause(self, after_key: str | None) -> str:
        start = pd.Timestamp(self.scope.window_start).strftime(SOCRATA_TIMESTAMP)
        end = pd.Timestamp(self.scope.window_end).strftime(SOCRATA_TIMESTAMP)
        parts = [
            _in_list("agency", self.scope.agencies),
            _in_list("complaint_type", self.scope.complaint_types),
            f"{self.date_field} >= {_quote(start)}",
            f"{self.date_field} < {_quote(end)}",
        ]
        if after_key is not None:
            parts.append(f"{self.key_field} > {_quote(after_key)}")
        return " AND ".join(parts)

    def fetch_page(self, after_key: str | None, limit: int) -> list[dict[str, Any]]:
        params = {
            "$select": ", ".join(self.columns),
            "$where": self.where_clause(after_key),
            "$order": self.key_field,
            "$limit": limit,
        }
        page = self.http.get_json(self.records_url, params)
        if not isinstance(page, list) or not all(isinstance(r, dict) for r in page):
            raise InvalidResponseError(f"{self.records_url}: expected a JSON list of records")
        return page
