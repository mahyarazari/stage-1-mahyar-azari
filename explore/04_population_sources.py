"""Explore candidate population reference sources and how well they cover 311 zips.

Exploration only: not part of the final package.
Answers three questions:
  1. What does each source actually look like (rows, totals, oddities)?
  2. Do the two sources agree on population?
  3. What share of HPD_top6 records can be joined to each source?

Outputs (explore/output/, git-ignored):
  modzcta_crosswalk.csv, acs_nyc.csv, zip_coverage.csv
"""
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path("explore/output")

MODZCTA_URL = "https://data.cityofnewyork.us/resource/pri4-ifjk.json"
ACS_URL = "https://api.census.gov/data/{year}/acs/acs5"
ACS_YEARS = [2024, 2023, 2022]          # try newest first, keep the first that exists
ACS_POP_VAR = "B01003_001E"
ACS_MISSING_SENTINEL = -666666666       # Census code for "not available"

SERVICE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
WINDOW = "created_date >= '2024-01-01T00:00:00' AND created_date < '2025-07-01T00:00:00'"
SCOPE = (
    "agency='HPD' AND complaint_type in('HEAT/HOT WATER', 'UNSANITARY CONDITION', "
    "'PLUMBING', 'PAINT/PLASTER', 'DOOR/WINDOW', 'WATER LEAK')"
)


def get_json(url, params=None):
    response = requests.get(url, params=params, timeout=300)
    # Print the URL without the key so it never ends up in logs or screenshots.
    print(f"  {response.status_code}  {response.url.split('&key=')[0][:110]}")
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "json" not in content_type:
        raise ValueError(
            f"Expected JSON but got '{content_type}' from {response.url.split('&key=')[0]} "
            f"(redirected: {bool(response.history)})"
        )
    return response.json()


# --------------------------------------------------------------- 1) MODZCTA
def load_modzcta():
    print("\n[1] MODZCTA (NYC Open Data)")
    try:
        rows = get_json(MODZCTA_URL, {"$select": "modzcta, label, zcta, pop_est", "$limit": 1000})
        modzcta = pd.DataFrame(rows)
    except requests.HTTPError:
        print("  $select failed, fetching all columns and dropping geometry")
        modzcta = pd.DataFrame(get_json(MODZCTA_URL, {"$limit": 1000}))
        modzcta = modzcta.drop(columns=[c for c in modzcta.columns if "geom" in c.lower()])
    print("  columns:", list(modzcta.columns))

    modzcta["pop_est"] = pd.to_numeric(modzcta["pop_est"], errors="coerce")
    print("  rows:", len(modzcta))
    print("  total pop_est:", int(modzcta["pop_est"].sum()))
    print("  rows with pop_est == 0 or missing:",
          int((modzcta["pop_est"].fillna(0) == 0).sum()))

    # The zcta column holds several zips in one string: explode it to one zip per row.
    crosswalk = modzcta.assign(zip=modzcta["zcta"].fillna("").str.split(",")).explode("zip")
    crosswalk["zip"] = crosswalk["zip"].str.strip()
    crosswalk = crosswalk[crosswalk["zip"] != ""][["zip", "modzcta", "pop_est"]]
    print("  zips in crosswalk:", crosswalk["zip"].nunique())

    repeated = crosswalk[crosswalk["zip"].duplicated(keep=False)].sort_values("zip")
    print("  zips mapped to more than one MODZCTA (join fan-out risk):", repeated["zip"].nunique())
    if not repeated.empty:
        print(repeated.head(10).to_string(index=False))
    return modzcta, crosswalk


# ------------------------------------------------------------------- 2) ACS
def load_acs(nyc_zips):
    print("\n[2] ACS 5-year (Census API)")
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        print("  CENSUS_API_KEY is not set: skipping ACS")
        return pd.DataFrame(columns=["zcta", "population"])

    for year in ACS_YEARS:
        try:
            rows = get_json(ACS_URL.format(year=year),
                            {"get": f"NAME,{ACS_POP_VAR}", "for": "zip code tabulation area:*",
                             "key": api_key})
        except (requests.HTTPError, ValueError) as error:
            print(f"  vintage {year} failed ({error}), trying older")
            continue
        acs = pd.DataFrame(rows[1:], columns=rows[0])
        acs = acs.rename(columns={"zip code tabulation area": "zcta", ACS_POP_VAR: "population"})
        acs["population"] = pd.to_numeric(acs["population"], errors="coerce")
        acs["vintage"] = year
        print(f"  using vintage {year}  ({year - 4}-{year} estimates), all US ZCTAs:", len(acs))

        sentinel = (acs["population"] == ACS_MISSING_SENTINEL).sum()
        print("  rows with the 'not available' sentinel:", int(sentinel))
        acs.loc[acs["population"] == ACS_MISSING_SENTINEL, "population"] = pd.NA

        nyc = acs[acs["zcta"].isin(nyc_zips)].copy()
        print("  NYC zips (from crosswalk) found in ACS:", len(nyc), "of", len(nyc_zips))
        print("  total NYC population (ACS, these ZCTAs):", int(nyc["population"].sum()))
        return nyc
    raise RuntimeError("No ACS vintage available")


# -------------------------------------------------------------- 3) coverage
def zip_coverage(crosswalk_zips, acs_zips):
    print("\n[3] Coverage of HPD_top6 records")
    rows = get_json(SERVICE_URL, {"$select": "incident_zip, count(*) AS n",
                                  "$where": f"{WINDOW} AND {SCOPE}",
                                  "$group": "incident_zip", "$limit": 5000})
    zips = pd.DataFrame(rows)
    zips["n"] = zips["n"].astype(int)
    zips["incident_zip"] = zips.get("incident_zip")
    total = zips["n"].sum()

    zips["in_modzcta"] = zips["incident_zip"].isin(crosswalk_zips)
    zips["in_acs"] = zips["incident_zip"].isin(acs_zips)
    print("  distinct incident_zip values:", len(zips), "| records:", total)
    print("  % records joinable to MODZCTA:", round(100 * zips.loc[zips["in_modzcta"], "n"].sum() / total, 3))
    print("  % records joinable to ACS:    ", round(100 * zips.loc[zips["in_acs"], "n"].sum() / total, 3))

    unmatched = zips[~zips["in_modzcta"]].sort_values("n", ascending=False)
    print("  top zips NOT in MODZCTA crosswalk:")
    print(unmatched.head(15).to_string(index=False))
    return zips


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("retrieved at:", datetime.now().isoformat(timespec="seconds"))

    modzcta, crosswalk = load_modzcta()
    nyc_zips = set(crosswalk["zip"])
    acs = load_acs(nyc_zips)
    coverage = zip_coverage(nyc_zips, set(acs["zcta"]))

    crosswalk.to_csv(OUTPUT_DIR / "modzcta_crosswalk.csv", index=False)
    acs.to_csv(OUTPUT_DIR / "acs_nyc.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "zip_coverage.csv", index=False)

    print("\n[4] Do the two sources agree?")
    print("  MODZCTA total:", int(modzcta["pop_est"].sum()))
    if acs.empty:
        print("  ACS total    : skipped")
    else:
        print("  ACS total    :", int(acs["population"].sum()))


if __name__ == "__main__":
    main()