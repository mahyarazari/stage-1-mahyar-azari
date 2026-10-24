"""Tests for the reference tables and the enrich stage. No network."""
import json
import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_factory.cleaning import CLEAN_COLUMNS, CLEANED_FILE
from data_factory.enrichment import AREAS_FILE, ENRICHED_FILE, enrich_chunk, enrich_dataset
from data_factory.exceptions import InvalidResponseError, MissingInputError, ReferenceDataError
from data_factory.references import (
    area_boroughs,
    fetch_references,
    parse_population,
    parse_zcta_county,
    reference_files,
)
from data_factory.tables import read_csv_text, write_csv

COUNTIES = {"36005": "BRONX", "36047": "BROOKLYN", "36061": "MANHATTAN",
            "36081": "QUEENS", "36085": "STATEN ISLAND"}
ALIASES = {"11249": "11211"}

CENSUS = "\n".join([
    "OID_ZCTA5_20|GEOID_ZCTA5_20|GEOID_COUNTY_20|AREALAND_PART",
    "1|10463|36005|9580",
    "2|10463|36061|420",      # Marble Hill: legally Manhattan
    "3|11211|36047|1000",
    "4|11249|36047|500",
    "5|11001|36081|116",
    "6|11001|36059|884",      # Nassau County: outside NYC
    "7||36047|50",            # county land in no ZCTA: blank key
    "8|07001|34023|999",      # New Jersey: not touching NYC
]) + "\n"

POPULATION = [
    {"modzcta": "10463", "zcta": "10463", "pop_est": "70000"},
    {"modzcta": "11211", "zcta": "11211, 11206", "pop_est": "100000"},
    {"modzcta": "99999", "zcta": "10041", "pop_est": "0"},
]


def references():
    return parse_population(POPULATION), parse_zcta_county(CENSUS, COUNTIES)


def case(key, zip_code="10463", borough="BRONX"):
    values = dict.fromkeys(CLEAN_COLUMNS)
    values.update(unique_key=str(key), incident_zip=zip_code, borough=borough)
    return values


def cases(*rows):
    return pd.DataFrame(list(rows), columns=CLEAN_COLUMNS).astype("string")


# ---------------------------------------------------------------- references
class PopulationReferenceTests(unittest.TestCase):
    def test_zip_lists_are_split_into_one_zip_per_area(self):
        population, _ = references()
        self.assertEqual(population.zip_to_modzcta["11206"], "11211")
        self.assertEqual(population.zero_population_areas, ["99999"])

    def test_zip_in_two_areas_is_refused(self):
        rows = POPULATION + [{"modzcta": "10001", "zcta": "10463", "pop_est": "5"}]
        with self.assertRaises(ReferenceDataError):
            parse_population(rows)

    def test_blank_area_key_is_refused(self):
        with self.assertRaises(ReferenceDataError):
            parse_population([{"modzcta": "", "zcta": "10463", "pop_est": "5"}])

    def test_negative_population_is_refused(self):
        with self.assertRaises(ReferenceDataError):
            parse_population([{"modzcta": "10463", "zcta": "10463", "pop_est": "-1"}])


class BoroughReferenceTests(unittest.TestCase):
    def test_boroughs_touched_by_each_zip(self):
        _, boroughs = references()
        self.assertEqual(boroughs.zip_boroughs["10463"], ("BRONX", "MANHATTAN"))
        self.assertEqual(boroughs.zip_boroughs["11001"], ("QUEENS",))   # outside part ignored
        self.assertNotIn("07001", boroughs.zip_boroughs)
        self.assertEqual(boroughs.split_zips, ["10463", "11001"])

    def test_blank_keys_are_dropped_and_counted(self):
        _, boroughs = references()
        self.assertEqual(boroughs.blank_key_rows, 1)
        self.assertNotIn("", boroughs.zip_boroughs)

    def test_missing_column_is_refused(self):
        with self.assertRaises(ReferenceDataError):
            parse_zcta_county("GEOID_ZCTA5_20|AREALAND_PART\n10463|5\n", COUNTIES)

    def test_area_borough_is_the_one_with_most_land(self):
        population, boroughs = references()
        result = area_boroughs(population, boroughs)
        self.assertEqual(result["10463"], "BRONX")
        self.assertEqual(result["11211"], "BROOKLYN")
        self.assertIsNone(result["99999"])          # its only zip is not in the Census file


class FakeHttp:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def get_json(self, url, params=None):
        self.calls.append(url)
        return self.rows

    def get_text(self, url, params=None):
        self.calls.append(url)
        return CENSUS


class FetchReferencesTests(unittest.TestCase):
    def test_references_are_downloaded_once_then_read_from_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            files = reference_files(Path(folder), "pop-url", "census-url")
            http = FakeHttp(POPULATION)
            fetch_references(http, files, "pop-url", "census-url")
            fetch_references(http, files, "pop-url", "census-url")
            self.assertEqual(http.calls, ["pop-url", "census-url"])
            self.assertEqual(files.meta()["population"]["url"], "pop-url")
            self.assertIn("retrieved_at_utc", files.meta()["zcta_county"])

    def test_a_full_page_of_population_rows_is_treated_as_truncated(self):
        with tempfile.TemporaryDirectory() as folder:
            files = reference_files(Path(folder), "pop-url", "census-url")
            with self.assertRaises(InvalidResponseError):
                fetch_references(FakeHttp([{}] * 1000), files, "pop-url", "census-url")


# ---------------------------------------------------------------- enrich
class EnrichChunkTests(unittest.TestCase):
    def enrich(self, *rows):
        population, boroughs = references()
        return enrich_chunk(cases(*rows), population, boroughs, ALIASES).set_index("unique_key")

    def test_alias_zip_gets_the_population_of_its_parent_area(self):
        out = self.enrich(case(1, zip_code="11249", borough="BROOKLYN")).loc["1"]
        self.assertEqual((out["incident_zip"], out["join_zip"], out["modzcta"]), ("11249", "11211", "11211"))
        self.assertEqual(out["population"], 100000)
        self.assertEqual(out["population_status"], "matched")

    def test_population_status_for_each_kind_of_miss(self):
        out = self.enrich(case(1, zip_code=None), case(2, zip_code="10803"), case(3, zip_code="10041"))
        self.assertEqual(out["population_status"].tolist(),
                         ["zip_missing", "zip_not_in_reference", "no_population"])
        self.assertTrue(out["population"].isna().all())

    def test_legal_borough_of_a_split_zip_is_consistent(self):
        out = self.enrich(case(1, borough="MANHATTAN"), case(2, borough="QUEENS"))
        self.assertEqual(out["borough_check"].tolist(), ["consistent", "inconsistent"])
        self.assertEqual(out.loc["1", "borough"], "MANHATTAN")         # never overwritten

    def test_missing_borough_is_filled_only_from_a_single_borough_zip(self):
        out = self.enrich(case(1, zip_code="11211", borough=None), case(2, zip_code="10463", borough=None))
        self.assertEqual((out.loc["1", "borough"], out.loc["1", "borough_source"]), ("BROOKLYN", "zip"))
        self.assertTrue(pd.isna(out.loc["2", "borough"]))
        self.assertEqual(out["borough_check"].tolist(), ["not_checkable", "not_checkable"])

    def test_missing_zip_never_matches_a_blank_reference_key(self):
        out = self.enrich(case(1, zip_code=None, borough=None)).loc["1"]
        self.assertTrue(pd.isna(out["modzcta"]))
        self.assertTrue(pd.isna(out["borough"]))


class EnrichDatasetTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        write_csv(self.work / CLEANED_FILE, cases(
            case(1), case(2, zip_code="11249", borough="BROOKLYN"),
            case(3, zip_code=None), case(4, borough="QUEENS")))

    def tearDown(self):
        self._tmp.cleanup()
        logging.disable(logging.NOTSET)

    def run_enrich(self):
        population, boroughs = references()
        return enrich_dataset(self.work, population, boroughs, ALIASES, chunk_rows=2)

    def test_row_count_is_unchanged_and_counted(self):
        stats = self.run_enrich()
        self.assertEqual((stats.rows_before_join, stats.rows_after_join), (4, 4))
        self.assertEqual(len(read_csv_text(self.work / ENRICHED_FILE)), 4)

    def test_counts_for_the_manifest(self):
        stats = self.run_enrich()
        self.assertEqual(stats.alias_applied, 1)
        self.assertEqual(stats.population_status, {"matched": 3, "zip_missing": 1})
        self.assertEqual(stats.borough_check["inconsistent"], 1)
        self.assertEqual(stats.inconsistent_pairs, {"10463|QUEENS": 1})
        saved = json.loads((self.work / "enrich_stats.json").read_text("utf-8"))
        self.assertEqual(saved["reference"]["blank_key_rows_dropped"], 1)

    def test_area_table_is_written_for_the_summary(self):
        self.run_enrich()
        areas = read_csv_text(self.work / AREAS_FILE).set_index("modzcta")
        self.assertEqual(areas.loc["10463", "borough"], "BRONX")

    def test_output_is_byte_identical_on_a_second_run(self):
        self.run_enrich()
        first = (self.work / ENRICHED_FILE).read_bytes()
        self.run_enrich()
        self.assertEqual(first, (self.work / ENRICHED_FILE).read_bytes())

    def test_missing_input_is_refused(self):
        (self.work / CLEANED_FILE).unlink()
        with self.assertRaises(MissingInputError):
            self.run_enrich()


if __name__ == "__main__":
    unittest.main()
