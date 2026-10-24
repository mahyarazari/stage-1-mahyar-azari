"""Tests for the data contract. No network: every record is built by hand."""
import unittest

import pandas as pd

from data_factory.contract import (
    REASON_COLUMN,
    ReasonCode,
    Scope,
    reason_counts,
    validate,
)

SCOPE = Scope(
    agencies=("HPD",),
    complaint_types=("HEAT/HOT WATER", "PLUMBING"),
    window_start=pd.Timestamp("2024-01-01"),
    window_end=pd.Timestamp("2025-07-01"),
)
FETCHED_AT = pd.Timestamp("2026-10-24")
PLACEHOLDERS = ("N/A", "NA", "00000")


def record(**overrides):
    """A healthy raw record, as the API sends it (all text). Override fields to break it."""
    base = {
        "unique_key": "60012345",
        "created_date": "2024-03-01T08:00:00.000",
        "closed_date": "2024-03-03T08:00:00.000",
        "status": "Closed",
        "agency": "HPD",
        "complaint_type": "HEAT/HOT WATER",
        "incident_zip": "10463",
        "borough": "BRONX",
    }
    base.update(overrides)
    return {key: value for key, value in base.items() if value is not None}


def run(*records):
    return validate(pd.DataFrame(list(records)), SCOPE, FETCHED_AT, PLACEHOLDERS)


class HealthyRecordTests(unittest.TestCase):
    def test_healthy_record_passes_with_parsed_dates(self):
        result = run(record())
        self.assertEqual(len(result.valid), 1)
        self.assertEqual(len(result.quarantined), 0)
        self.assertEqual(result.valid["created_date"].iloc[0], pd.Timestamp("2024-03-01 08:00"))

    def test_open_case_without_closed_date_passes(self):
        # The API omits empty fields: an open case simply has no closed_date key.
        result = run(record(closed_date=None, status="Open"))
        self.assertEqual(len(result.valid), 1)
        self.assertTrue(pd.isna(result.valid["closed_date"].iloc[0]))

    def test_missing_optional_field_is_not_an_error(self):
        result = run(record(incident_zip=None, borough=None))
        self.assertEqual(len(result.quarantined), 0)


class QuarantineTests(unittest.TestCase):
    def assert_quarantined_for(self, broken_record, reason):
        result = run(broken_record)
        self.assertEqual(len(result.valid), 0)
        self.assertEqual(len(result.quarantined), 1)
        self.assertIn(reason.value, result.quarantined[REASON_COLUMN].iloc[0].split("|"))

    def test_closed_before_created(self):
        self.assert_quarantined_for(
            record(closed_date="2024-02-28T08:00:00.000"), ReasonCode.CLOSED_BEFORE_CREATED)

    def test_closed_after_download_time(self):
        self.assert_quarantined_for(
            record(closed_date="2027-01-01T00:00:00.000"), ReasonCode.CLOSED_IN_FUTURE)

    def test_missing_required_field(self):
        self.assert_quarantined_for(record(unique_key=None), ReasonCode.MISSING_REQUIRED)

    def test_non_numeric_key(self):
        self.assert_quarantined_for(record(unique_key="60A12"), ReasonCode.INVALID_KEY)

    def test_unparseable_date(self):
        self.assert_quarantined_for(record(created_date="03/01/2024"), ReasonCode.UNPARSEABLE_DATE)

    def test_out_of_scope_complaint_type(self):
        self.assert_quarantined_for(record(complaint_type="NOISE"), ReasonCode.OUT_OF_SCOPE)

    def test_unknown_borough(self):
        self.assert_quarantined_for(record(borough="JERSEY"), ReasonCode.BOROUGH_INVALID)

    def test_quarantine_keeps_original_raw_values(self):
        result = run(record(unique_key="  60A12 "))
        self.assertEqual(result.quarantined["unique_key"].iloc[0], "  60A12 ")


class EdgeCaseTests(unittest.TestCase):
    def test_window_start_is_inclusive_and_end_is_exclusive(self):
        result = run(
            record(unique_key="1", created_date="2024-01-01T00:00:00.000", closed_date=None),
            record(unique_key="2", created_date="2025-07-01T00:00:00.000", closed_date=None),
        )
        self.assertEqual(result.valid["unique_key"].tolist(), ["1"])
        self.assertEqual(result.quarantined[REASON_COLUMN].tolist(), [ReasonCode.OUT_OF_WINDOW.value])

    def test_zero_duration_is_not_quarantined(self):
        # Suspicious but not impossible: flagged later, not rejected here.
        result = run(record(closed_date="2024-03-01T08:00:00.000"))
        self.assertEqual(len(result.valid), 1)

    def test_zip_placeholders_and_spaces_become_missing(self):
        result = run(
            record(unique_key="1", incident_zip="N/A"),
            record(unique_key="2", incident_zip="00000"),
            record(unique_key="3", incident_zip="   "),
            record(unique_key="4", incident_zip=" 10463 "),
            record(unique_key="5", incident_zip="10463-1234"),
        )
        self.assertEqual(len(result.quarantined), 0)
        zips = result.valid.set_index("unique_key")["incident_zip"]
        self.assertTrue(zips[["1", "2", "3"]].isna().all())
        self.assertEqual(zips[["4", "5"]].tolist(), ["10463", "10463"])

    def test_malformed_zip_is_quarantined(self):
        result = run(record(incident_zip="1046"))
        self.assertEqual(result.quarantined[REASON_COLUMN].iloc[0], ReasonCode.ZIP_MALFORMED.value)

    def test_unknown_but_well_formed_zip_passes(self):
        # Membership in a reference table is not part of the contract (references can be outdated).
        result = run(record(incident_zip="12345"))
        self.assertEqual(len(result.valid), 1)

    def test_several_reasons_are_all_recorded_in_fixed_order(self):
        result = run(record(borough="JERSEY", closed_date="2024-02-01T00:00:00.000"))
        self.assertEqual(
            result.quarantined[REASON_COLUMN].iloc[0],
            "CLOSED_BEFORE_CREATED|BOROUGH_INVALID",
        )


class ReasonCountTests(unittest.TestCase):
    def test_each_reason_of_a_record_is_counted(self):
        result = run(
            record(unique_key="1", borough="JERSEY", closed_date="2024-02-01T00:00:00.000"),
            record(unique_key="2", borough="JERSEY"),
            record(unique_key="3"),
        )
        counts = reason_counts(result.quarantined)
        self.assertEqual(counts[ReasonCode.BOROUGH_INVALID.value], 2)
        self.assertEqual(counts[ReasonCode.CLOSED_BEFORE_CREATED.value], 1)
        self.assertEqual(counts[ReasonCode.OUT_OF_SCOPE.value], 0)

    def test_no_quarantine_gives_all_zero_counts(self):
        counts = reason_counts(run(record()).quarantined)
        self.assertEqual(set(counts.values()), {0})
        self.assertEqual(set(counts), {code.value for code in ReasonCode})


if __name__ == "__main__":
    unittest.main()
