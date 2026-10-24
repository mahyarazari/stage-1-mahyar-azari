"""Tests for the analysis helpers behind the notebook. No network."""
import logging
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_factory import analysis
from data_factory.contract import Scope
from data_factory.raw_store import RawPageStore
from data_factory.references import parse_zcta_county

SCOPE = Scope(("HPD",), ("PLUMBING",), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-07-01"))
COUNTIES = {"36005": "BRONX", "36047": "BROOKLYN", "36061": "MANHATTAN", "36081": "QUEENS", "36085": "STATEN ISLAND"}


def cases(rows):
    frame = pd.DataFrame(rows)
    for column, kind in analysis.CASE_COLUMNS.items():
        if column not in frame:
            frame[column] = None
    frame["is_open"] = frame["is_open"].fillna(False).astype(bool)
    frame["is_duplicate"] = frame["is_duplicate"].fillna(False).astype(bool)
    return frame


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_duplicates_are_left_out_of_every_table(self):
        data = cases([{"complaint_type": "PLUMBING", "days_to_close": 2.0},
                      {"complaint_type": "PLUMBING", "days_to_close": 2.0, "is_duplicate": True},
                      {"complaint_type": "PLUMBING", "days_to_close": 4.0}])
        table = analysis.duration_summary(data)
        self.assertEqual(table.loc["PLUMBING", "problems"], 2)
        self.assertEqual(table.loc["PLUMBING", "median"], 3.0)

    def test_concentration_finds_the_top_address(self):
        data = cases([{"incident_address": "A"}] * 8 + [{"incident_address": "B"}, {"incident_address": "C"}])
        _, facts = analysis.concentration(data)
        self.assertEqual((facts["top_address"], facts["top_address_problems"]), ("A", 8))
        self.assertEqual(facts["top_1pct_share_pct"], 80.0)

    def test_year_over_year_pairs_the_same_months(self):
        table = pd.DataFrame({"borough": ["BRONX"], "complaint_type": ["PLUMBING"],
                              "2024-01": [10.0], "2024-02": [10.0], "2025-01": [15.0], "2025-02": [15.0]})
        result = analysis.year_over_year(table)
        self.assertEqual(result["change_pct"].iloc[0], 50.0)

    def test_open_cases_count_as_slowest_in_monthly_medians(self):
        data = cases([{"complaint_type": "PLUMBING", "created_month": "2024-01", "days_to_close": 1.0},
                      {"complaint_type": "PLUMBING", "created_month": "2024-01", "days_to_close": 3.0},
                      {"complaint_type": "PLUMBING", "created_month": "2024-01", "is_open": True},
                      {"complaint_type": "HEAT/HOT WATER", "created_month": "2024-01", "days_to_close": 1.0}])
        monthly, _ = analysis.heat_versus_others(data, "HEAT/HOT WATER")
        self.assertEqual(monthly.loc["2024-01", "other_types_median_days"], 3.0)

    def test_borough_rules_naive_versus_legal(self):
        census = "GEOID_ZCTA5_20|GEOID_COUNTY_20|AREALAND_PART\n10463|36005|9580\n10463|36061|420\n"
        boroughs = parse_zcta_county(census, COUNTIES)
        data = cases([{"incident_zip": "10463", "borough": "MANHATTAN"},
                      {"incident_zip": "10463", "borough": "BRONX"}])
        result = analysis.borough_rule_comparison(data, boroughs)
        self.assertEqual((result["largest_area_rule_disagreements"], result["legal_rule_disagreements"]), (1, 0))

    def test_quarantine_demo_catches_exactly_the_broken_records(self):
        page = [{"unique_key": f"{k:08d}", "created_date": "2024-03-01T08:00:00.000",
                 "closed_date": "2024-03-02T08:00:00.000", "agency": "HPD", "complaint_type": "PLUMBING"}
                for k in range(1, 41)]
        with tempfile.TemporaryDirectory() as folder:
            store = RawPageStore(Path(folder))
            store.write_page(1, page)
            store.mark_complete({"fetched_at_utc": "2026-10-24T12:00:00"})
            demo = analysis.quarantine_demo(store, SCOPE, ("N/A",), max_ratio=0.01)
        self.assertEqual(demo["small_corruption"]["quarantined"], 20)
        self.assertTrue(demo["small_corruption"]["exactly_the_broken_ones"])
        self.assertTrue(demo["wide_corruption"]["run_stopped"])
        self.assertFalse(demo["wide_corruption"]["valid_file_published"])

    def test_memory_profile_reads_every_page(self):
        with tempfile.TemporaryDirectory() as folder:
            store = RawPageStore(Path(folder))
            for number in (1, 2, 3):
                store.write_page(number, [{"unique_key": f"{number}{k:04d}"} for k in range(100)])
            profile = analysis.raw_memory_profile([store])
        self.assertEqual((profile["pages"], profile["records"]), (3, 300))
        self.assertFalse(math.isnan(profile["peak_mb_reading_all_pages"]))


if __name__ == "__main__":
    unittest.main()
