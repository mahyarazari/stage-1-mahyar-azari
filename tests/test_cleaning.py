"""Tests for the clean stage. No network: validated records are written by hand."""
import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_factory.cleaning import CLEANED_FILE, clean_chunk, clean_dataset
from data_factory.config import CleanConfig
from data_factory.contract import SOURCE_COLUMNS
from data_factory.exceptions import MissingInputError
from data_factory.tables import read_csv_text, write_csv
from data_factory.validation import VALIDATED_FILE

AS_OF = pd.Timestamp("2026-10-01")
SETTINGS = CleanConfig(chunk_rows=2, unspecified_values=("Unspecified", "0 Unspecified"),
                       latitude_range=(40.47, 40.93), longitude_range=(-74.27, -73.68))


def row(key, **overrides):
    """A validated record (text, dates already in the fixed format)."""
    base = {"unique_key": str(key), "created_date": "2024-03-01T08:00:00",
            "closed_date": "2024-03-03T20:00:00", "status": "Closed", "agency": "HPD",
            "complaint_type": "PLUMBING", "descriptor": "LEAK", "incident_zip": "10463",
            "incident_address": "1 MAIN STREET", "borough": "BRONX",
            "community_board": "08 BRONX", "latitude": "40.88", "longitude": "-73.90",
            "open_data_channel_type": "PHONE", "resolution_action_updated_date": None}
    base.update(overrides)
    return base


def frame(*rows):
    return pd.DataFrame(list(rows)).reindex(columns=list(SOURCE_COLUMNS)).astype("string")


class CleanChunkTests(unittest.TestCase):
    def clean_one(self, **overrides):
        return clean_chunk(frame(row(1, **overrides)), AS_OF, SETTINGS).iloc[0]

    def test_days_to_close_and_month(self):
        cleaned = self.clean_one()
        self.assertAlmostEqual(cleaned["days_to_close"], 2.5)
        self.assertEqual(cleaned["created_month"], "2024-03")
        self.assertFalse(cleaned["is_open"])

    def test_open_case_has_no_duration(self):
        cleaned = self.clean_one(closed_date=None, status="Open")
        self.assertTrue(cleaned["is_open"])
        self.assertTrue(pd.isna(cleaned["days_to_close"]))

    def test_case_closed_after_as_of_counts_as_open(self):
        cleaned = self.clean_one(closed_date="2026-10-15T09:00:00")
        self.assertTrue(cleaned["is_open"])
        self.assertTrue(pd.isna(cleaned["closed_at"]))

    def test_case_closed_exactly_at_as_of_is_closed(self):
        self.assertFalse(self.clean_one(closed_date="2026-10-01T00:00:00")["is_open"])

    def test_month_follows_creation_not_closing(self):
        cleaned = self.clean_one(created_date="2024-01-31T23:59:59", closed_date="2024-02-01T00:00:01")
        self.assertEqual(cleaned["created_month"], "2024-01")

    def test_text_is_trimmed_and_spaces_collapsed(self):
        cleaned = self.clean_one(incident_address="  10 east  116   street ")
        self.assertEqual(cleaned["incident_address"], "10 EAST 116 STREET")

    def test_unspecified_values_become_missing(self):
        cleaned = self.clean_one(borough="Unspecified", community_board="0 Unspecified")
        self.assertTrue(pd.isna(cleaned["borough"]))
        self.assertTrue(pd.isna(cleaned["community_board"]))

    def test_coordinates_outside_the_city_are_removed_and_flagged(self):
        cleaned = self.clean_one(latitude="0", longitude="0")
        self.assertTrue(pd.isna(cleaned["latitude"]))
        self.assertTrue(cleaned["flag_coords_invalid"])

    def test_missing_coordinates_are_not_flagged(self):
        self.assertFalse(self.clean_one(latitude=None, longitude=None)["flag_coords_invalid"])

    def test_flags_for_zero_duration_midnight_and_status_conflict(self):
        cleaned = self.clean_one(created_date="2024-03-01T00:00:00",
                                 closed_date="2024-03-01T00:00:00", status="Assigned")
        self.assertTrue(cleaned["flag_zero_duration"])
        self.assertTrue(cleaned["flag_created_midnight"])
        self.assertTrue(cleaned["flag_status_conflict"])

    def test_open_case_with_open_status_is_not_a_conflict(self):
        self.assertFalse(self.clean_one(closed_date=None, status="Open")["flag_status_conflict"])


class CleanDatasetTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        logging.disable(logging.NOTSET)

    def run_clean(self, *rows):
        write_csv(self.work / VALIDATED_FILE, frame(*rows))
        stats = clean_dataset(self.work, AS_OF, SETTINGS)
        return stats, read_csv_text(self.work / CLEANED_FILE).set_index("unique_key")

    def test_duplicates_across_chunks_keep_the_smallest_key(self):
        # chunk_rows=2: the three twins land in two different chunks.
        stats, out = self.run_clean(row(60000003), row(60000009, descriptor="WALL"),
                                    row(60000001), row(60000002))
        self.assertEqual(out.loc["60000001", "is_duplicate"], "False")
        self.assertEqual(out.loc["60000002", "is_duplicate"], "True")
        self.assertEqual(out.loc["60000003", "is_duplicate"], "True")
        self.assertEqual(out.loc["60000009", "is_duplicate"], "False")   # other descriptor
        self.assertEqual((stats.duplicate_groups, stats.duplicate_rows), (1, 2))
        self.assertEqual(out.loc["60000001", "duplicate_group_id"], out.loc["60000003", "duplicate_group_id"])

    def test_different_closing_time_means_a_different_problem(self):
        _, out = self.run_clean(row(1), row(2, closed_date="2024-12-05T10:00:00"))
        self.assertEqual(out["is_duplicate"].tolist(), ["False", "False"])

    def test_two_open_twins_are_one_problem(self):
        _, out = self.run_clean(row(1, closed_date=None), row(2, closed_date=None))
        self.assertEqual(out["is_duplicate"].tolist(), ["False", "True"])

    def test_rows_without_descriptor_are_never_grouped(self):
        _, out = self.run_clean(row(1, descriptor=None), row(2, descriptor=None))
        self.assertEqual(out["is_duplicate"].tolist(), ["False", "False"])
        self.assertTrue(out["duplicate_group_id"].isna().all())

    def test_address_spelling_differences_still_group(self):
        _, out = self.run_clean(row(1, incident_address="10 EAST  116 STREET"),
                                row(2, incident_address="10 east 116 street"))
        self.assertEqual(out["is_duplicate"].tolist(), ["False", "True"])

    def test_output_is_byte_identical_on_a_second_run(self):
        rows = [row(1), row(2), row(3, closed_date=None), row(4, latitude="0")]
        self.run_clean(*rows)
        first = (self.work / CLEANED_FILE).read_bytes()
        self.run_clean(*rows)
        self.assertEqual(first, (self.work / CLEANED_FILE).read_bytes())

    def test_missing_input_is_refused(self):
        with self.assertRaises(MissingInputError):
            clean_dataset(self.work, AS_OF, SETTINGS)


if __name__ == "__main__":
    unittest.main()
