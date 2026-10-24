"""Tests for the moving window (--since-last-run). No network.

The base snapshot and the references are prepared on disk (as in test_manifest); a fake
source serves the records of the new period and remembers the window it was asked for.
"today" is fixed by replacing pipeline.today_utc.
"""
import json
import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd

from data_factory import pipeline
from data_factory.config import load_config
from data_factory.exceptions import MissingInputError
from data_factory.manifest import MANIFEST_FILE
from data_factory.publish import summary_file
from data_factory.references import reference_files
from tests.test_manifest import CENSUS, POPULATION, PROJECT_CONFIG, raw_record

NEW_RECORDS = [
    {**raw_record(10), "created_date": "2025-07-15T09:00:00.000", "closed_date": "2025-07-20T09:00:00.000"},
    {**raw_record(11), "created_date": "2025-08-20T09:00:00.000", "closed_date": "2025-08-21T09:00:00.000"},
]


class FakeIncrementSource:
    """Serves NEW_RECORDS once, in key order, and records the scope it was built with."""

    key_field = "unique_key"
    scopes = []

    def __init__(self, http, records_url, metadata_url, scope, columns):
        FakeIncrementSource.scopes.append(scope)

    def check_schema(self, expected):
        pass

    def fetch_page(self, after_key, limit):
        return [r for r in NEW_RECORDS if after_key is None or r["unique_key"] > after_key][:limit]


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        FakeIncrementSource.scopes = []
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "factory.toml").write_text(PROJECT_CONFIG.read_text("utf-8"), "utf-8")
        self.config = load_config(root / "factory.toml")
        files = reference_files(self.config.paths.raw / "references", self.config.references.population_url,
                                self.config.references.zcta_county_url)
        for path, text in ((files.population, json.dumps(POPULATION)), (files.zcta_county, CENSUS)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, "utf-8")
            path.with_name(path.name + ".meta.json").write_text('{"url": "u", "retrieved_at_utc": "t"}', "utf-8")

    def tearDown(self):
        self._tmp.cleanup()
        logging.disable(logging.NOTSET)

    def make_base(self):
        store = pipeline.raw_store(self.config)
        store.write_page(1, [raw_record(1), raw_record(2)])
        store.mark_complete({"fetched_at_utc": "2026-10-24T12:00:00", "pages": 1, "records": 2})

    def fetch_increment(self, today):
        with mock.patch.object(pipeline, "SocrataSource", FakeIncrementSource), \
             mock.patch.object(pipeline, "today_utc", return_value=today):
            pipeline.fetch(pipeline.RunContext(config=self.config, since_last_run=True))

    def test_needs_a_complete_base_snapshot(self):
        with self.assertRaises(MissingInputError):
            self.fetch_increment(datetime(2025, 9, 3))

    def test_increment_starts_where_the_data_ends_and_stops_before_yesterday(self):
        self.make_base()
        self.fetch_increment(datetime(2025, 9, 3))
        scope = FakeIncrementSource.scopes[0]
        self.assertEqual((scope.window_start, scope.window_end),
                         (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-02")))
        segments = pipeline.chain(self.config).segments()
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1].store.directory.name, "20250701T000000_20250902T000000")
        self.assertEqual(segments[1].store.count_records(), 2)

    def test_a_second_run_the_same_day_fetches_nothing(self):
        self.make_base()
        self.fetch_increment(datetime(2025, 9, 3))
        self.fetch_increment(datetime(2025, 9, 3))
        self.assertEqual(len(FakeIncrementSource.scopes), 1)
        self.assertEqual(len(pipeline.chain(self.config).segments()), 2)

    def test_next_increment_continues_without_gap(self):
        self.make_base()
        self.fetch_increment(datetime(2025, 9, 3))
        self.fetch_increment(datetime(2025, 9, 10))
        self.assertEqual(FakeIncrementSource.scopes[1].window_start, pd.Timestamp("2025-09-02"))
        self.assertEqual(pipeline.chain(self.config).window_end(), pd.Timestamp("2025-09-09"))

    def test_interrupted_increment_is_resumed_with_its_own_window(self):
        self.make_base()
        unfinished = pipeline.chain(self.config).increment(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-01"))
        unfinished.store.write_page(1, [NEW_RECORDS[0]])
        self.fetch_increment(datetime(2025, 9, 3))
        scope = FakeIncrementSource.scopes[0]
        self.assertEqual(scope.window_end, pd.Timestamp("2025-08-01"))   # not today's cutoff
        self.assertTrue(unfinished.store.is_complete())

    def test_as_of_moves_with_the_data_but_never_backwards(self):
        self.make_base()
        self.assertEqual(pipeline.effective_as_of(self.config), self.config.as_of)
        self.fetch_increment(datetime(2026, 10, 30))
        self.assertEqual(pipeline.effective_as_of(self.config), pd.Timestamp("2026-10-29"))

    def test_whole_factory_uses_base_and_increment_together(self):
        self.make_base()
        self.fetch_increment(datetime(2025, 9, 3))
        with mock.patch.object(pipeline, "SocrataSource", FakeIncrementSource), \
             mock.patch.object(pipeline, "today_utc", return_value=datetime(2025, 9, 3)):
            pipeline.run(pipeline.RunContext(config=self.config),
                         ["validate", "clean", "enrich", "publish"])
        manifest = json.loads((self.config.paths.curated / MANIFEST_FILE).read_text("utf-8"))
        self.assertEqual(manifest["counts"]["raw_records"], 4)
        self.assertEqual(manifest["scope"]["window_end_exclusive"], "2025-09-02T00:00:00")
        self.assertEqual(len(manifest["source"]["segments"]), 2)
        table = pd.read_csv(self.config.paths.curated / summary_file("borough", "cases"), dtype=str)
        self.assertEqual(list(table.columns)[-3:], ["2025-07", "2025-08", "2025-09"])


if __name__ == "__main__":
    unittest.main()
