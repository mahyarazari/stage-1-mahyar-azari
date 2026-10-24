"""Build an independent zip -> borough mapping and compare it with the 311 borough column.

Exploration only: not part of the final package.
Source: Census 2020 ZCTA-to-County relationship file (pipe-delimited, no API key needed).
A NYC borough is a county, so county FIPS -> borough is a fixed, documented table.

Questions answered:
  1. How many NYC ZCTAs span more than one county (ambiguous borough)?
  2. For HPD_top6 records, how often does our borough agree with the 311 borough column?
  3. Where they disagree, which zips and which borough pairs?

Outputs (explore/output/, git-ignored):
  zip_borough_mapping.csv, borough_comparison.csv
"""
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path("explore/output")

RELATIONSHIP_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/"
    "tab20_zcta520_county20_natl.txt"
)

# County FIPS (state 36 = New York) -> borough name as written in the 311 data.
COUNTY_TO_BOROUGH = {
    "36005": "BRONX",
    "36047": "BROOKLYN",
    "36061": "MANHATTAN",
    "36081": "QUEENS",
    "36085": "STATEN ISLAND",
}

ZIP_ALIASES = {"11249": "11211"}   # same idea as the population join; see decisions.md

SERVICE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
WINDOW = "created_date >= '2024-01-01T00:00:00' AND created_date < '2025-07-01T00:00:00'"
SCOPE = (
    "agency='HPD' AND complaint_type in('HEAT/HOT WATER', 'UNSANITARY CONDITION', "
    "'PLUMBING', 'PAINT/PLASTER', 'DOOR/WINDOW', 'WATER LEAK')"
)


def find_column(columns, *must_contain):
    """Pick the one column whose name contains every given fragment (robust to layout tweaks)."""
    matches = [c for c in columns if all(part in c.upper() for part in must_contain)]
    if len(matches) != 1:
        raise ValueError(f"Expected one column with {must_contain}, found {matches}")
    return matches[0]


# ------------------------------------------------- 1) independent mapping
def build_mapping():
    print("[1] Census ZCTA -> county relationship file")
    response = requests.get(RELATIONSHIP_URL, timeout=300)
    print(f"  {response.status_code}  {RELATIONSHIP_URL}")
    response.raise_for_status()
    print(f"  size: {len(response.content) / 1e6:.1f} MB")

    rel = pd.read_csv(io.StringIO(response.content.decode("utf-8-sig")), sep="|", dtype=str)
    print("  columns:", list(rel.columns))

    zcta_col = find_column(rel.columns, "GEOID", "ZCTA")
    county_col = find_column(rel.columns, "GEOID", "COUNTY")
    rel = rel.rename(columns={zcta_col: "zcta", county_col: "county"})
    rel["land_part"] = pd.to_numeric(rel["AREALAND_PART"], errors="coerce").fillna(0)

    # Keep every piece of any ZCTA that touches NYC, including pieces outside NYC.
    nyc_zctas = rel.loc[rel["county"].isin(COUNTY_TO_BOROUGH), "zcta"].unique()
    pieces = rel[rel["zcta"].isin(nyc_zctas)].copy()
    pieces["borough"] = pieces["county"].map(COUNTY_TO_BOROUGH).fillna("OUTSIDE_NYC")
    pieces["land_share"] = pieces["land_part"] / pieces.groupby("zcta")["land_part"].transform("sum")
    print("  ZCTAs touching NYC:", len(nyc_zctas))

    multi = pieces.groupby("zcta").filter(lambda g: len(g) > 1).sort_values(["zcta", "land_share"])
    print("  ZCTAs spanning more than one county:", multi["zcta"].nunique())
    if not multi.empty:
        print(multi[["zcta", "borough", "land_share"]].round(3).to_string(index=False))

    # Rule (to be justified): the county holding the largest land share wins.
    mapping = (pieces.sort_values(["zcta", "land_share"], ascending=[True, False])
                     .drop_duplicates("zcta")[["zcta", "borough", "land_share"]]
                     .rename(columns={"land_share": "winning_share"}))
    return mapping


# ------------------------------------------------- 2) compare with 311
def compare_with_311(mapping):
    print("\n[2] Compare with the 311 borough column (HPD_top6)")
    rows = requests.get(SERVICE_URL, timeout=300, params={
        "$select": "incident_zip, borough, count(*) AS n",
        "$where": f"{WINDOW} AND {SCOPE}",
        "$group": "incident_zip, borough",
        "$limit": 10000,
    })
    rows.raise_for_status()
    df = pd.DataFrame(rows.json())
    df["n"] = df["n"].astype(int)
    for col in ("incident_zip", "borough"):
        if col not in df:
            df[col] = pd.NA

    df["join_zip"] = df["incident_zip"].replace(ZIP_ALIASES)
    df = df.merge(mapping, how="left", left_on="join_zip", right_on="zcta")
    df = df.rename(columns={"borough_x": "borough_311", "borough_y": "borough_ours"})

    def classify(row):
        if pd.isna(row["borough_ours"]):
            return "no_mapping"
        if pd.isna(row["borough_311"]) or row["borough_311"] == "Unspecified":
            return "311_unspecified"
        return "agree" if row["borough_311"] == row["borough_ours"] else "disagree"

    df["result"] = df.apply(classify, axis=1)
    total = df["n"].sum()
    summary = df.groupby("result")["n"].sum().sort_values(ascending=False)
    print("  records:", total)
    print((100 * summary / total).round(3).rename("pct_records").to_string())

    disagree = df[df["result"] == "disagree"].sort_values("n", ascending=False)
    print("\n  top disagreements:")
    print(disagree[["incident_zip", "borough_311", "borough_ours", "winning_share", "n"]]
          .head(20).to_string(index=False))

    print("\n  disagreements by borough pair:")
    print(disagree.groupby(["borough_311", "borough_ours"])["n"].sum()
          .sort_values(ascending=False).to_string())
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("retrieved at:", datetime.now().isoformat(timespec="seconds"))
    mapping = build_mapping()
    comparison = compare_with_311(mapping)
    mapping.to_csv(OUTPUT_DIR / "zip_borough_mapping.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "borough_comparison.csv", index=False)


if __name__ == "__main__":
    main()