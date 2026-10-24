"""Tests for the raw store, the Socrata source and the resumable, idempotent download.

No network: a FakeSource serves pages from an in-memory list of records, in key order,
and can be told to fail on a given call to simulate an interrupted run.
"""
import json
import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_factory.contract import Scope
from data_factory.exceptions import (
    DataValidationError,
    InvalidResponseError,
    NetworkError,
    SourceSchemaChangedError,
)
from data_factory.ingest import fetch_snapshot, snapshot_id
from data_factory.raw_store import RawPageStore
from data_factory.sources import SocrataSource

EXPECTED = {"unique_key": "text", "created_date": "calendar_date"}
NOW = datetime(2026, 10, 24, 12, 0, 0)
SCOPE = Scope(agencies=("HPD",), complaint_types=("PLUMBING",),
              window_start=pd.Timestamp("2024-01-01"), window_end=pd.Timestamp("2025-07-01"))


def records(count):
    return [{"unique_key": f"{key:08d}", "created_date": "2024-03-01T08:00:00.000"}
            for key in range(1, count + 1)]


class FakeSource:
    key_field = "unique_key"

    def __init__(self, all_records, fail_on_call=None, schema=None):
        self.all_records = all_records
        self.fail_on_call = fail_on_call
        self.schema = schema if schema is not None else dict(EXPECTED)
        self.calls = []          # after_key of every page request

    def check_schema(self, expected):
        missing = [name for name in expected if name not in self.schema]
        if missing:
            raise SourceSchemaChangedError(f"missing {missing}")

    def fetch_page(self, after_key, limit):
        self.calls.append(after_key)
        if self.fail_on_call == len(self.calls):
            raise NetworkError("connection lost")
        newer = [r for r in self.all_records if after_key is None or r["unique_key"] > after_key]
        return newer[:limit]


class TempFolderMixin:
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.store = RawPageStore(Path(self._tmp.name) / "snapshot")

    def tearDown(self):
        self._tmp.cleanup()
        logging.disable(logging.NOTSET)

    def stored_keys(self):
        return [r["unique_key"] for page in self.store.read_pages() for r in page]


# ---------------------------------------------------------------- raw store
class RawStoreTests(TempFolderMixin, unittest.TestCase):
    def test_empty_store_starts_from_the_beginning(self):
        self.assertIsNone(self.store.last_key())
        self.assertEqual(self.store.next_page_number(), 1)

    def test_resume_point_is_last_key_of_last_complete_page(self):
        self.store.write_page(1, records(3))
        self.store.write_page(2, records(5)[3:])
        self.assertEqual(self.store.last_key(), "00000005")
        self.assertEqual(self.store.next_page_number(), 3)

    def test_unfinished_temporary_file_is_ignored(self):
        self.store.write_page(1, records(2))
        (self.store.directory / "page-00002.jsonl.tmp").write_text('{"unique_key": "000', "utf-8")
        self.assertEqual(self.store.last_key(), "00000002")
        self.assertEqual(self.store.next_page_number(), 2)

    def test_complete_snapshot_cannot_be_changed(self):
        self.store.write_page(1, records(1))
        self.store.mark_complete({"records": 1})
        with self.assertRaises(DataValidationError):
            self.store.write_page(2, records(2))

    def test_pages_are_written_with_lf_line_endings(self):
        path = self.store.write_page(1, records(2))
        self.assertNotIn(b"\r\n", path.read_bytes())


# ---------------------------------------------------------------- ingest
class IngestTests(TempFolderMixin, unittest.TestCase):
    def test_full_download_in_pages(self):
        source = FakeSource(records(10))
        result = fetch_snapshot(source, self.store, EXPECTED, page_size=4, now=NOW)
        self.assertEqual((result.pages_written, result.records_written), (3, 10))
        self.assertEqual(self.stored_keys(), [f"{k:08d}" for k in range(1, 11)])
        self.assertEqual(self.store.completion_info()["records"], 10)

    def test_short_last_page_avoids_an_extra_empty_request(self):
        source = FakeSource(records(10))
        fetch_snapshot(source, self.store, EXPECTED, page_size=4, now=NOW)
        self.assertEqual(len(source.calls), 3)   # 4 + 4 + 2, no fourth request

    def test_exact_multiple_needs_one_empty_request(self):
        source = FakeSource(records(8))
        fetch_snapshot(source, self.store, EXPECTED, page_size=4, now=NOW)
        self.assertEqual(len(source.calls), 3)   # 4 + 4 + empty answer

    def test_interrupted_run_resumes_after_last_complete_page(self):
        with self.assertRaises(NetworkError):
            fetch_snapshot(FakeSource(records(10), fail_on_call=3), self.store, EXPECTED,
                           page_size=4, now=NOW)
        self.assertFalse(self.store.is_complete())
        self.assertEqual(self.store.last_key(), "00000008")

        second = FakeSource(records(10))
        fetch_snapshot(second, self.store, EXPECTED, page_size=4, now=NOW)
        self.assertEqual(second.calls, ["00000008"])        # did not start from scratch
        keys = self.stored_keys()
        self.assertEqual(keys, [f"{k:08d}" for k in range(1, 11)])
        self.assertEqual(len(keys), len(set(keys)))          # no duplicate, no gap

    def test_complete_snapshot_is_not_downloaded_again(self):
        fetch_snapshot(FakeSource(records(5)), self.store, EXPECTED, page_size=4, now=NOW)
        before = [p.read_bytes() for p in self.store.page_files()]
        again = FakeSource(records(7))                        # the live source has changed
        result = fetch_snapshot(again, self.store, EXPECTED, page_size=4, now=NOW)
        self.assertTrue(result.already_complete)
        self.assertEqual(again.calls, [])
        self.assertEqual([p.read_bytes() for p in self.store.page_files()], before)

    def test_leftover_temporary_file_is_removed_on_resume(self):
        self.store.write_page(1, records(4))
        leftover = self.store.directory / "page-00002.jsonl.tmp"
        leftover.write_text("partial", "utf-8")
        fetch_snapshot(FakeSource(records(6)), self.store, EXPECTED, page_size=4, now=NOW)
        self.assertFalse(leftover.exists())
        self.assertEqual(len(self.stored_keys()), 6)

    def test_keys_out_of_order_stop_the_download(self):
        shuffled = records(4)
        shuffled[1], shuffled[2] = shuffled[2], shuffled[1]
        source = FakeSource(shuffled)
        source.fetch_page = lambda after_key, limit: shuffled
        with self.assertRaises(InvalidResponseError):
            fetch_snapshot(source, self.store, EXPECTED, page_size=10, now=NOW)
        self.assertEqual(self.store.page_files(), [])

    def test_schema_change_stops_before_any_download(self):
        source = FakeSource(records(3), schema={"unique_key": "text"})
        with self.assertRaises(SourceSchemaChangedError):
            fetch_snapshot(source, self.store, EXPECTED, page_size=4, now=NOW)
        self.assertEqual(source.calls, [])


class SnapshotIdTests(unittest.TestCase):
    def test_same_request_same_folder_and_any_change_a_new_one(self):
        first = snapshot_id("u", ["b", "a"], SCOPE)
        self.assertEqual(first, snapshot_id("u", ["a", "b"], SCOPE))
        wider = Scope(SCOPE.agencies, SCOPE.complaint_types, SCOPE.window_start,
                      pd.Timestamp("2025-08-01"))
        self.assertNotEqual(first, snapshot_id("u", ["a", "b"], wider))


# ---------------------------------------------------------------- Socrata source
class FakeHttp:
    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    def get_json(self, url, params=None):
        self.requests.append((url, params))
        return self.answer


class SocrataSourceTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def source(self, answer):
        http = FakeHttp(answer)
        return SocrataSource(http, "records", "metadata", SCOPE, ["unique_key", "created_date"]), http

    def test_page_request_uses_keyset_order_and_scope(self):
        source, http = self.source([])
        source.fetch_page("00000042", 1000)
        params = http.requests[0][1]
        self.assertEqual(params["$order"], "unique_key")
        self.assertEqual(params["$limit"], 1000)
        self.assertEqual(params["$select"], "unique_key, created_date")
        where = params["$where"]
        self.assertIn("unique_key > '00000042'", where)
        self.assertIn("agency in('HPD')", where)
        self.assertIn("created_date >= '2024-01-01T00:00:00'", where)
        self.assertIn("created_date < '2025-07-01T00:00:00'", where)

    def test_first_page_has_no_key_condition(self):
        source, _ = self.source([])
        self.assertNotIn("unique_key >", source.where_clause(None))

    def test_quotes_inside_values_are_escaped(self):
        source, _ = self.source([])
        source.scope = Scope(("HPD",), ("O'BRIEN",), SCOPE.window_start, SCOPE.window_end)
        self.assertIn("'O''BRIEN'", source.where_clause(None))

    def test_answer_that_is_not_a_list_of_records_is_invalid(self):
        source, _ = self.source({"error": "oops"})
        with self.assertRaises(InvalidResponseError):
            source.fetch_page(None, 10)

    def test_schema_check_reports_missing_and_changed_columns(self):
        metadata = {"columns": [{"fieldName": "unique_key", "dataTypeName": "number"}]}
        source, _ = self.source(metadata)
        with self.assertRaises(SourceSchemaChangedError) as caught:
            source.check_schema(EXPECTED)
        message = str(caught.exception)
        self.assertIn("created_date", message)
        self.assertIn("unique_key: text -> number", message)

    def test_schema_check_accepts_extra_columns(self):
        metadata = {"columns": [{"fieldName": "unique_key", "dataTypeName": "text"},
                                {"fieldName": "created_date", "dataTypeName": "calendar_date"},
                                {"fieldName": "new_column", "dataTypeName": "text"}]}
        source, _ = self.source(metadata)
        source.check_schema(EXPECTED)   # no error


if __name__ == "__main__":
    unittest.main()
