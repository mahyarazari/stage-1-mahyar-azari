"""Measure possible double submissions (same complaint, different unique_key).

Exploration only: not part of the final package.
Downloads every HPD_top6 record of a few sample months, then counts how many records
would be called duplicates under several candidate definitions.

Outputs (explore/output/, git-ignored): duplicate_groups.csv (examples to inspect by eye)
"""
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path("explore/output")
SERVICE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
SAMPLE_MONTHS = ["2024-01", "2024-07"]   # one heating-season month, one summer month
PAGE_SIZE = 50_000
SCOPE = (
    "agency='HPD' AND complaint_type in('HEAT/HOT WATER', 'UNSANITARY CONDITION', "
    "'PLUMBING', 'PAINT/PLASTER', 'DOOR/WINDOW', 'WATER LEAK')"
)
COLUMNS = "unique_key, created_date, closed_date, status, complaint_type, descriptor, incident_address, bbl"

# Candidate definitions: records sharing all these values (and with different keys) are "duplicates".
DEFINITIONS = {
    "A_address_type_exact_time": ["incident_address", "complaint_type", "created_date"],
    "B_address_type_descriptor_exact_time": ["incident_address", "complaint_type", "descriptor", "created_date"],
    "C_building_type_descriptor_exact_time": ["bbl", "complaint_type", "descriptor", "created_date"],
    "D_address_type_descriptor_same_day": ["incident_address", "complaint_type", "descriptor", "created_day"],
}


def fetch_month(month):
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    where = (f"{SCOPE} AND created_date >= '{start:%Y-%m-%dT%H:%M:%S}' "
             f"AND created_date < '{end:%Y-%m-%dT%H:%M:%S}'")
    frames, offset = [], 0
    while True:
        response = requests.get(SERVICE_URL, timeout=300, params={
            "$select": COLUMNS, "$where": where, "$order": "unique_key",
            "$limit": PAGE_SIZE, "$offset": offset,
        })
        response.raise_for_status()
        page = pd.DataFrame(response.json())
        print(f"  {month} offset {offset}: {len(page)} rows")
        if page.empty:
            break
        frames.append(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return pd.concat(frames, ignore_index=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([fetch_month(m) for m in SAMPLE_MONTHS], ignore_index=True)
    for column in ["incident_address", "bbl", "descriptor"]:
        if column not in df:
            df[column] = pd.NA
    created = pd.to_datetime(df["created_date"])
    df["created_day"] = created.dt.date.astype(str)

    print("\nrecords in sample:", len(df))
    print("same unique_key twice (technical duplicates):", int(df["unique_key"].duplicated().sum()))
    print("% created exactly at 00:00:00 (date-only timestamps):",
          round(100 * (created.dt.normalize() == created).mean(), 2))
    print("% records missing address:", round(100 * df["incident_address"].isna().mean(), 2))
    print("% records missing bbl:", round(100 * df["bbl"].isna().mean(), 2))

    print("\nrecords that have at least one 'twin' under each definition")
    examples = []
    for name, keys in DEFINITIONS.items():
        usable = df.dropna(subset=keys)           # a missing key value never makes a twin
        size = usable.groupby(keys)["unique_key"].transform("count")
        twins = usable[size > 1]
        extra = len(twins) - twins.groupby(keys).ngroups   # records that would be removed
        print(f"  {name:42} groups: {twins.groupby(keys).ngroups:6}  "
              f"records in groups: {len(twins):6}  would remove: {extra:6} "
              f"({100 * extra / len(df):.2f}%)")
        examples.append(twins.sort_values(keys).head(30).assign(definition=name))

    pd.concat(examples).to_csv(OUTPUT_DIR / "duplicate_groups.csv", index=False)
    print("\nexamples saved to explore/output/duplicate_groups.csv")


if __name__ == "__main__":
    main()
