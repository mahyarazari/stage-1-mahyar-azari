"""Tests for the publish stage. No network: enriched cases are written by hand."""
import logging
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_factory.config import AggregateConfig
from data_factory.contract import Scope
from data_factory.enrichment import AREAS_FILE, ENRICHED_FILE, ENRICH_COLUMNS
from data_factory.exceptions import MissingInputError
from data_factory.publish import (
    CASES_FILE,
    QUALITY_FILE,
    _median_with_open,
    publish,
    summary_file,
    window_months,
)
from data_factory.tables import read_csv_text, write_csv

SCOPE = Scope(("HPD",), ("PLUMBING", "HEAT/HOT WATER"),
              pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01"))
SETTINGS = AggregateConfig(levels=("borough", "modzcta"), min_cases_per_cell=2, rate_per=10000)


def case(key, days, month="2024-01", borough="BRONX", area="10463", kind="PLUMBING",
         is_open=False, duplicate=False):
    values = dict.fromkeys(ENRICH_COLUMNS)
    values.update(unique_key=str(key), created_month=month, borough=borough, modzcta=area,
                  complaint_type=kind, days_to_close=None if is_open else days,
                  is_open=is_open, is_duplicate=duplicate, latitude=40.8)
    return values


class SummaryHelpersTests(unittest.TestCase):
    def test_months_cover_the_whole_window_end_excluded(self):
        self.assertEqual(window_months(SCOPE), ["2024-01", "2024-02", "2024-03"])

    def test_open_cases_count_as_longer_than_closed_ones(self):
        # Dropping the open case would give a median of 2; keeping it as "infinite" gives 3.
        self.assertEqual(_median_with_open(pd.Series([1.0, 3.0, math.inf])), 3.0)

    def test_median_is_unknown_when_most_cases_are_open(self):
        self.assertTrue(math.isnan(_median_with_open(pd.Series([1.0, math.inf, math.inf]))))


class PublishTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.work, self.curated = root / "work", root / "curated"
        write_csv(self.work / AREAS_FILE, pd.DataFrame(
            {"modzcta": ["10463", "11211"], "population": [20000, 50000], "borough": ["BRONX", "BROOKLYN"]}))

    def tearDown(self):
        self._tmp.cleanup()
        logging.disable(logging.NOTSET)

    def run_publish(self, *cases):
        write_csv(self.work / ENRICHED_FILE, pd.DataFrame(list(cases), columns=ENRICH_COLUMNS))
        return publish(self.work, self.curated, SCOPE, SETTINGS)

    def table(self, level, metric):
        return read_csv_text(self.curated / summary_file(level, metric))

    def test_wide_table_has_one_column_per_month(self):
        self.run_publish(case(1, 2.0), case(2, 4.0), case(3, 1.0, month="2024-03"))
        table = self.table("borough", "cases")
        self.assertEqual(list(table.columns), ["borough", "complaint_type", "2024-01", "2024-02", "2024-03"])
        self.assertEqual(table.iloc[0][["2024-01", "2024-02", "2024-03"]].tolist(), ["2", "0", "1"])

    def test_duplicates_are_counted_once(self):
        self.run_publish(case(1, 2.0), case(2, 2.0, duplicate=True), case(3, 4.0))
        self.assertEqual(self.table("borough", "cases").iloc[0]["2024-01"], "2")

    def test_median_keeps_open_cases_as_slowest(self):
        self.run_publish(case(1, 1.0), case(2, 3.0), case(3, None, is_open=True))
        self.assertEqual(float(self.table("borough", "median_days").iloc[0]["2024-01"]), 3.0)

    def test_rate_per_residents(self):
        self.run_publish(case(1, 1.0), case(2, 2.0))          # 2 problems, 20,000 residents
        self.assertEqual(float(self.table("modzcta", "rate").iloc[0]["2024-01"]), 1.0)

    def test_borough_population_is_the_sum_of_its_areas(self):
        self.run_publish(*[case(k, 1.0, borough="BROOKLYN", area="11211") for k in range(5)])
        self.assertEqual(float(self.table("borough", "rate").iloc[0]["2024-01"]), 1.0)

    def test_small_cells_show_count_but_hide_median_and_rate(self):
        stats = self.run_publish(case(1, 1.0))                 # 1 case < min_cases_per_cell 2
        self.assertEqual(self.table("borough", "cases").iloc[0]["2024-01"], "1")
        self.assertTrue(pd.isna(self.table("borough", "median_days").iloc[0]["2024-01"]))
        self.assertTrue(pd.isna(self.table("borough", "rate").iloc[0]["2024-01"]))
        self.assertEqual(stats.cells_hidden_small["borough"], 1)

    def test_case_without_area_is_left_out_of_that_level_only(self):
        stats = self.run_publish(case(1, 1.0, area=None), case(2, 2.0, area=None))
        self.assertEqual(stats.cases_without_area, {"borough": 0, "modzcta": 2})

    def test_case_file_and_quality_report_are_published(self):
        self.run_publish(case(1, 1.0), case(2, 5.0, borough=None))
        self.assertEqual(len(read_csv_text(self.curated / CASES_FILE)), 2)
        report = read_csv_text(self.curated / QUALITY_FILE).set_index("column")
        self.assertEqual(report.loc["borough", "missing"], "1")
        self.assertEqual(report.loc["borough", "missing_pct"], "50")
        self.assertEqual((report.loc["days_to_close", "min"], report.loc["days_to_close", "max"]), ("1", "5"))
        self.assertEqual(report.loc["unique_key", "distinct"], "2")

    def test_outputs_are_byte_identical_on_a_second_run(self):
        cases = [case(1, 1.0), case(2, 3.0, kind="HEAT/HOT WATER"), case(3, None, is_open=True)]
        self.run_publish(*cases)
        first = {p.name: p.read_bytes() for p in self.curated.iterdir()}
        self.run_publish(*cases)
        self.assertEqual(first, {p.name: p.read_bytes() for p in self.curated.iterdir()})

    def test_missing_input_is_refused(self):
        with self.assertRaises(MissingInputError):
            publish(self.work, self.curated, SCOPE, SETTINGS)


if __name__ == "__main__":
    unittest.main()
