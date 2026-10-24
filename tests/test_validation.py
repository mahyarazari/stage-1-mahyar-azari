"""Tests for the validate stage and the atomic CSV writer. No network."""
import json
import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_factory.contract import Scope
from data_factory.exceptions import MissingInputError, QuarantineThresholdError
from data_factory.raw_store import RawPageStore
from data_factory.tables import AtomicCsvWriter, read_csv_text
from data_factory.validation import (
    QUARANTINE_FILE,
    STATS_FILE,
    VALIDATED_FILE,
    validate_snapshot,
)

SCOPE = Scope(agencies=("HPD",), complaint_types=("PLUMBING",),
              window_start=pd.Timestamp("2024-01-01"), window_end=pd.Timestamp("2025-07-01"))
PLACEHOLDERS = ("N/A", "00000")


def record(key, **overrides):
    base = {"unique_key": f"{key:08d}", "created_date": "2024-03-01T08:00:00.000",
            "closed_date": "2024-03-02T08:00:00.000", "agency": "HPD",
            "complaint_type": "PLUMBING", "incident_zip": "10463", "borough": "BRONX"}
    base.update(overrides)
    return base


class ValidateStageTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.store = RawPageStore(root / "raw" / "snap")
        self.work = root / "work" / "snap"
        self.quarantine = root / "quarantine" / "snap"

    def tearDown(self):
        self._tmp.cleanup()
        logging.disable(logging.NOTSET)

    def snapshot(self, *pages):
        for number, page in enumerate(pages, start=1):
            self.store.write_page(number, page)
        self.store.mark_complete({"fetched_at_utc": "2026-10-24T12:00:00", "pages": len(pages)})

    def run_stage(self, max_ratio=0.5):
        return validate_snapshot(self.store, self.work, self.quarantine, SCOPE, PLACEHOLDERS, max_ratio)

    def test_valid_and_quarantined_records_go_to_separate_files(self):
        self.snapshot([record(1), record(2, closed_date="2024-01-01T00:00:00.000")],
                      [record(3, incident_zip="N/A"), record(4, borough="JERSEY")])
        stats = self.run_stage()
        self.assertEqual((stats.raw_records, stats.valid_records, stats.quarantined_records), (4, 2, 2))
        valid = read_csv_text(self.work / VALIDATED_FILE)
        bad = read_csv_text(self.quarantine / QUARANTINE_FILE)
        self.assertEqual(valid["unique_key"].tolist(), ["00000001", "00000003"])
        self.assertEqual(bad["reason_codes"].tolist(), ["CLOSED_BEFORE_CREATED", "BOROUGH_INVALID"])
        self.assertTrue(pd.isna(valid["incident_zip"].iloc[1]))   # placeholder became missing

    def test_dates_are_written_in_one_fixed_format(self):
        self.snapshot([record(1)])
        self.run_stage()
        valid = read_csv_text(self.work / VALIDATED_FILE)
        self.assertEqual(valid["created_date"].iloc[0], "2024-03-01T08:00:00")

    def test_counts_per_reason_are_saved_for_the_manifest(self):
        self.snapshot([record(1, borough="JERSEY"), record(2, borough="MARS"), record(3)])
        self.run_stage(max_ratio=0.9)
        saved = json.loads((self.work / STATS_FILE).read_text("utf-8"))
        self.assertEqual(saved["reasons"]["BOROUGH_INVALID"], 2)
        self.assertEqual(saved["reasons"]["OUT_OF_SCOPE"], 0)
        self.assertEqual(saved["raw_records"], 3)

    def test_threshold_exceeded_stops_without_publishing_valid_records(self):
        self.snapshot([record(1, borough="JERSEY"), record(2)])
        with self.assertRaises(QuarantineThresholdError):
            self.run_stage(max_ratio=0.1)
        self.assertTrue((self.quarantine / QUARANTINE_FILE).exists())   # evidence is kept
        self.assertFalse((self.work / VALIDATED_FILE).exists())
        self.assertEqual(list(self.work.glob("*.tmp")), [])               # nothing half-written

    def test_threshold_is_a_strict_upper_limit(self):
        self.snapshot([record(1, borough="JERSEY"), record(2)])
        stats = self.run_stage(max_ratio=0.5)                             # exactly 50%: allowed
        self.assertEqual(stats.quarantine_ratio, 0.5)

    def test_incomplete_snapshot_is_refused(self):
        self.store.write_page(1, [record(1)])
        with self.assertRaises(MissingInputError):
            self.run_stage()

    def test_second_run_gives_byte_identical_files(self):
        self.snapshot([record(1), record(2, borough="JERSEY")], [record(3)])
        self.run_stage()
        first = [(self.work / VALIDATED_FILE).read_bytes(), (self.quarantine / QUARANTINE_FILE).read_bytes()]
        self.run_stage()
        second = [(self.work / VALIDATED_FILE).read_bytes(), (self.quarantine / QUARANTINE_FILE).read_bytes()]
        self.assertEqual(first, second)
        self.assertNotIn(b"\r\n", first[0])


class AtomicCsvWriterTests(unittest.TestCase):
    def test_failure_before_commit_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.csv"
            with self.assertRaises(RuntimeError):
                with AtomicCsvWriter(path, ["a"]) as writer:
                    writer.append(pd.DataFrame({"a": [1]}))
                    raise RuntimeError("crash in the middle")
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
