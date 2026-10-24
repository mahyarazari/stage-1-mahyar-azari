"""Tests for loading and validating the configuration file."""
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_factory.config import load_config
from data_factory.exceptions import ConfigError, ExitCode

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "factory.toml"


class ConfigFileMixin:
    """Each test writes a modified copy of the real configuration into a temporary folder."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.original = PROJECT_CONFIG.read_text(encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, text):
        path = self.folder / "factory.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def assert_config_error(self, text, fragment):
        with self.assertRaises(ConfigError) as caught:
            load_config(self.write(text))
        self.assertIn(fragment, str(caught.exception))


class ProjectConfigTests(unittest.TestCase):
    def test_project_configuration_is_valid(self):
        config = load_config(PROJECT_CONFIG)
        self.assertEqual(config.scope.agencies, ("HPD",))
        self.assertEqual(len(config.scope.complaint_types), 6)
        self.assertEqual(config.scope.window_start, pd.Timestamp("2024-01-01"))
        self.assertEqual(config.zip.aliases, {"11249": "11211"})


class PathTests(ConfigFileMixin, unittest.TestCase):
    def test_relative_paths_follow_the_config_file_not_the_working_folder(self):
        path = self.write(self.original)
        elsewhere = tempfile.TemporaryDirectory()
        previous = os.getcwd()
        try:
            os.chdir(elsewhere.name)
            config = load_config(path)
        finally:
            os.chdir(previous)
            elsewhere.cleanup()
        self.assertEqual(config.paths.raw, (self.folder / "data" / "raw").resolve())


class InvalidConfigTests(ConfigFileMixin, unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(ConfigError):
            load_config(self.folder / "nope.toml")

    def test_invalid_toml(self):
        self.assert_config_error("[scope\n", "not valid TOML")

    def test_missing_section(self):
        start = self.original.index("[quarantine]")
        end = self.original.index("[aggregate]")
        self.assert_config_error(self.original[:start] + self.original[end:], "[quarantine]")

    def test_window_must_be_in_order(self):
        text = self.original.replace("window_end = 2025-07-01T00:00:00", "window_end = 2023-07-01T00:00:00")
        self.assert_config_error(text, "must be before")

    def test_as_of_cannot_be_inside_the_window(self):
        text = self.original.replace("as_of = 2026-10-01T00:00:00", "as_of = 2025-01-01T00:00:00")
        self.assert_config_error(text, "as_of")

    def test_page_size_must_be_positive(self):
        self.assert_config_error(self.original.replace("page_size = 50000", "page_size = 0"), "page_size")

    def test_wrong_type_is_reported(self):
        self.assert_config_error(self.original.replace("page_size = 50000", 'page_size = "big"'),
                                 "must be int")

    def test_alias_must_be_two_zips(self):
        self.assert_config_error(self.original.replace('"11249" = "11211"', '"11249" = "Brooklyn"'),
                                 "5-digit")

    def test_quarantine_ratio_must_be_a_share(self):
        self.assert_config_error(self.original.replace("max_ratio = 0.01", "max_ratio = 5"), "max_ratio")

    def test_unknown_aggregate_level(self):
        self.assert_config_error(self.original.replace('levels = ["borough", "modzcta"]',
                                                       'levels = ["borough", "planet"]'), "planet")

    def test_config_error_has_its_own_exit_code(self):
        self.assertEqual(ConfigError("x").exit_code, ExitCode.CONFIG)


if __name__ == "__main__":
    unittest.main()
