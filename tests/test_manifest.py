"""Tests for the run manifest, and an offline end-to-end run of the whole factory.

The end-to-end test builds a small project in a temporary folder: a complete raw snapshot
and cached references, so no network is needed (the schema check is replaced by a fake
source). It runs every stage twice and checks that the second manifest records identical
output hashes: the determinism acceptance criterion, as an automated test.
"""
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_factory import pipeline
from data_factory.config import load_config
from data_factory.manifest import MANIFEST_FILE, compare_with_previous, describe_outputs, file_sha256
from data_factory.references import reference_files

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "factory.toml"
CENSUS = "GEOID_ZCTA5_20|GEOID_COUNTY_20|AREALAND_PART\n10463|36005|9580\n10463|36061|420\n11211|36047|1000\n"
POPULATION = [{"modzcta": "10463", "zcta": "10463", "pop_est": "70000"},
              {"modzcta": "11211", "zcta": "11211", "pop_est": "100000"}]


def raw_record(key, zip_code="10463", borough="BRONX", closed="2024-03-03T08:00:00.000"):
    record = {"unique_key": f"{key:08d}", "created_date": "2024-03-01T08:00:00.000",
              "status": "Closed", "agency": "HPD", "complaint_type": "PLUMBING",
              "descriptor": "LEAK", "incident_zip": zip_code, "incident_address": f"{key} MAIN STREET",
              "borough": borough, "latitude": "40.88", "longitude": "-73.90"}
    if closed:
        record["closed_date"] = closed
    return record


class FakeSource:
    """Stands in for SocrataSource: the snapshot is already complete, only the schema is checked."""

    key_field = "unique_key"

    def __init__(self, *args, **kwargs):
        pass

    def check_schema(self, expected):
        pass


class ManifestPieceTests(unittest.TestCase):
    def test_outputs_are_described_by_hash_size_and_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.csv"
            path.write_bytes(b"x\n1\n2\n")
            described = describe_outputs({"a.csv": path, "missing.csv": Path(folder) / "no.csv"})
            self.assertEqual(list(described), ["a.csv"])
            self.assertEqual(described["a.csv"]["rows"], 2)
            self.assertEqual(described["a.csv"]["sha256"], file_sha256(path))

    def test_comparison_lists_changed_files(self):
        old = {"status": "succeeded", "run": {"started_at_utc": "t0"},
               "inputs": {"snapshot": "s", "config_sha256": "c", "git_commit": "g"},
               "outputs": {"a": {"sha256": "1"}, "b": {"sha256": "2"}}}
        new = {"inputs": dict(old["inputs"]), "outputs": {"a": {"sha256": "1"}, "b": {"sha256": "9"}}}
        result = compare_with_previous(old, new)
        self.assertEqual((result["same_inputs"], result["identical_outputs"], result["changed"]),
                         (True, False, ["b"]))

    def test_a_failed_previous_run_is_not_compared(self):
        result = compare_with_previous({"status": "failed"}, {"inputs": {}, "outputs": {}})
        self.assertIsNone(result["identical_outputs"])


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "factory.toml").write_text(PROJECT_CONFIG.read_text("utf-8"), "utf-8")
        self.config = load_config(root / "factory.toml")

        store = pipeline.raw_store(self.config)
        store.write_page(1, [raw_record(1), raw_record(2), raw_record(3, zip_code="11249", borough="BROOKLYN")])
        store.write_page(2, [raw_record(4, closed=None), raw_record(5, zip_code="N/A")])
        store.mark_complete({"fetched_at_utc": "2026-10-24T12:00:00", "pages": 2, "records": 5})

        files = reference_files(self.config.paths.raw / "references", self.config.references.population_url,
                                self.config.references.zcta_county_url)
        for path, text, url in ((files.population, json.dumps(POPULATION), "p"), (files.zcta_county, CENSUS, "c")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, "utf-8")
            path.with_name(path.name + ".meta.json").write_text(
                json.dumps({"url": url, "retrieved_at_utc": "2026-10-24T12:00:00"}), "utf-8")

    def tearDown(self):
        self._tmp.cleanup()
        logging.disable(logging.NOTSET)

    def run_factory(self):
        with mock.patch.object(pipeline, "SocrataSource", FakeSource):
            pipeline.run(pipeline.RunContext(config=self.config), list(pipeline.STAGE_NAMES))
        return json.loads((self.config.paths.curated / MANIFEST_FILE).read_text("utf-8"))

    def test_whole_factory_runs_and_manifest_has_the_required_fields(self):
        manifest = self.run_factory()
        self.assertEqual(manifest["status"], "succeeded")
        self.assertEqual(manifest["scope"]["window_start"], "2024-01-01T00:00:00")
        self.assertEqual((manifest["counts"]["raw_records"], manifest["counts"]["quarantined"]), (5, 0))
        self.assertIn("quarantine_by_reason", manifest["counts"])
        self.assertIn("started_at_utc", manifest["run"])
        self.assertIn("finished_at_utc", manifest["run"])
        self.assertEqual(manifest["code"]["package"], "0.1.0")
        self.assertIn("curated/cases.csv", manifest["outputs"])
        self.assertEqual(manifest["outputs"]["curated/cases.csv"]["rows"], 5)
        self.assertEqual(manifest["requests"]["this_run"]["calls"], 0)

    def test_second_run_has_identical_output_hashes(self):
        self.run_factory()
        second = self.run_factory()
        check = second["determinism"]
        self.assertTrue(check["same_inputs"])
        self.assertTrue(check["identical_outputs"], check["changed"])

    def test_failed_run_is_kept_in_history_but_not_published(self):
        self.run_factory()
        with mock.patch.object(pipeline, "SocrataSource", FakeSource), \
             mock.patch.dict(pipeline.STAGES, {"clean": mock.Mock(side_effect=RuntimeError("boom"))}):
            with self.assertRaises(RuntimeError):
                pipeline.run(pipeline.RunContext(config=self.config), list(pipeline.STAGE_NAMES))
        history = [p.name for p in (self.config.paths.work / "manifests").iterdir()]
        self.assertEqual(sum(name.endswith("_succeeded.json") for name in history), 1)
        self.assertEqual(sum(name.endswith("_failed.json") for name in history), 1)
        published = json.loads((self.config.paths.curated / MANIFEST_FILE).read_text("utf-8"))
        self.assertEqual(published["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
