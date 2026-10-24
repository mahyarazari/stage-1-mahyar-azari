"""Measure every candidate scope with identical criteria to choose the project scope.

Exploration only: not part of the final package.
- Exact numbers come from the API (server-side counts and groupings).
- Duration and zip-pattern statistics come from a small multi-month sample.

Outputs (in explore/output/, which is git-ignored):
- summary.csv            one row per candidate, all criteria side by side
- monthly_<name>.csv     records and open percentage per month, per candidate
"""
import math
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
OUTPUT_DIR = Path("explore/output")

# Wide range on purpose: shows month by month how "unsettled" recent data is.
RANGE_START = "2024-01-01T00:00:00"
RANGE_END = "2026-10-01T00:00:00"

# The 18-month window being evaluated (change it and re-run to compare windows).
WINDOW_START = "2024-01-01T00:00:00"
WINDOW_END = "2025-07-01T00:00:00"

# Months used for the duration sample (spread over seasons inside the window).
SAMPLE_MONTHS = ["2024-01", "2024-04", "2024-07", "2024-10", "2025-01", "2025-04"]
SAMPLE_PER_MONTH = 1000

MIN_RECORDS = 300_000
BATCH_SIZES = [1_000, 10_000, 50_000]

CANDIDATES = {
    "HPD_all": "agency='HPD'",
    "HPD_top6": (
        "agency='HPD' AND complaint_type in('HEAT/HOT WATER', 'UNSANITARY CONDITION', "
        "'PLUMBING', 'PAINT/PLASTER', 'DOOR/WINDOW', 'WATER LEAK')"
    ),
    "DSNY_all": "agency='DSNY'",
    "NYPD_noise": "agency='NYPD' AND starts_with(complaint_type, 'Noise')",
    "noise_all_agencies": "starts_with(complaint_type, 'Noise')",
}

MONTH_EXPR = "date_trunc_ym(created_date)"


# ---------------------------------------------------------------- HTTP helpers
def ask(params, retries=4):
    """GET with a simple wait-and-retry on rate limit (429) and server errors (5xx)."""
    for attempt in range(1, retries + 1):
        response = requests.get(BASE_URL, params=params, timeout=300)
        if response.status_code == 429 or response.status_code >= 500:
            wait = 15 * attempt
            print(f"    status {response.status_code}, retry {attempt} in {wait}s")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"Failed after {retries} attempts: {params}")


def count(where):
    return int(ask({"$select": "count(*) AS n", "$where": where})[0]["n"])


def window_where(scope):
    return f"({scope}) AND created_date >= '{WINDOW_START}' AND created_date < '{WINDOW_END}'"


# ------------------------------------------------------------ exact measurements
def monthly_profile(scope):
    """Records and still-open records per month over the wide range."""
    where = f"({scope}) AND created_date >= '{RANGE_START}' AND created_date < '{RANGE_END}'"
    base = {"$select": f"{MONTH_EXPR} AS month, count(*) AS n",
            "$group": MONTH_EXPR, "$order": MONTH_EXPR, "$limit": 100}
    total = pd.DataFrame(ask({**base, "$where": where}))
    still_open = pd.DataFrame(ask({**base, "$where": where + " AND closed_date IS NULL"}))

    profile = total.rename(columns={"n": "records"})
    profile["records"] = profile["records"].astype(int)
    if still_open.empty:
        profile["open"] = 0
    else:
        still_open = still_open.rename(columns={"n": "open"})
        profile = profile.merge(still_open, on="month", how="left")
        profile["open"] = profile["open"].fillna(0).astype(int)
    profile["pct_open"] = (100 * profile["open"] / profile["records"]).round(2)
    profile["month"] = profile["month"].str[:7]
    return profile


def exact_window_metrics(scope):
    where = window_where(scope)
    total = count(where)

    types = pd.DataFrame(ask({"$select": "complaint_type, count(*) AS n", "$where": where,
                              "$group": "complaint_type", "$order": "n DESC", "$limit": 500}))
    types["n"] = types["n"].astype(int)

    boroughs = pd.DataFrame(ask({"$select": "borough, count(*) AS n", "$where": where,
                                 "$group": "borough", "$limit": 50}))
    boroughs["n"] = boroughs["n"].astype(int)
    unspecified_borough = boroughs.loc[
        boroughs.get("borough", pd.Series(dtype=str)).isna()
        | (boroughs.get("borough") == "Unspecified"), "n"].sum()

    metrics = {
        "total": total,
        "margin_over_min_pct": round(100 * (total - MIN_RECORDS) / MIN_RECORDS, 1),
        "n_types": len(types),
        "top3_types_share_pct": round(100 * types["n"].head(3).sum() / total, 1),
        "pct_open": round(100 * count(where + " AND closed_date IS NULL") / total, 2),
        "n_closed_status_no_date": count(where + " AND status = 'Closed' AND closed_date IS NULL"),
        "n_not_closed_with_date": count(where + " AND status != 'Closed' AND closed_date IS NOT NULL"),
        "pct_zip_missing": round(100 * count(where + " AND incident_zip IS NULL") / total, 2),
        "pct_borough_unspecified": round(100 * unspecified_borough / total, 2),
    }
    for size in BATCH_SIZES:
        metrics[f"requests_at_{size}"] = math.ceil(total / size)
    return metrics


# ------------------------------------------------------------ sample measurements
def sample(scope):
    frames = []
    for month in SAMPLE_MONTHS:
        where = f"({scope}) AND {MONTH_EXPR} = '{month}-01T00:00:00'"
        frames.append(pd.DataFrame(ask({"$where": where, "$order": "unique_key",
                                        "$limit": SAMPLE_PER_MONTH})))
    return pd.concat(frames, ignore_index=True)


def sample_metrics(df):
    created = pd.to_datetime(df["created_date"])
    closed = pd.to_datetime(df["closed_date"]) if "closed_date" in df else pd.Series(pd.NaT, index=df.index)
    hours = (closed - created).dt.total_seconds() / 3600
    valid_hours = hours.dropna()

    zips = df["incident_zip"] if "incident_zip" in df else pd.Series("", index=df.index)
    zips = zips.fillna("").astype(str)

    return {
        "sample_size": len(df),
        "median_hours": round(valid_hours.median(), 1),
        "p90_hours": round(valid_hours.quantile(0.90), 1),
        "p99_days": round(valid_hours.quantile(0.99) / 24, 1),
        "pct_under_1h": round(100 * (valid_hours < 1).mean(), 1),
        "pct_zero_duration": round(100 * (valid_hours == 0).mean(), 2),
        "pct_negative_duration": round(100 * (valid_hours < 0).mean(), 2),
        "pct_over_365_days": round(100 * (valid_hours > 365 * 24).mean(), 2),
        "pct_zip_bad_pattern": round(100 * (~zips.str.fullmatch(r"1\d{4}")).mean(), 2),
        "pct_zip_with_spaces": round(100 * (zips != zips.str.strip()).mean(), 2),
        "pct_dup_unique_key_in_sample": round(100 * df["unique_key"].duplicated().mean(), 2),
    }


# ------------------------------------------------------------------------ main
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, scope in CANDIDATES.items():
        print(f"\n=== {name} ===")
        started = time.time()

        print("  monthly profile ...")
        profile = monthly_profile(scope)
        profile.to_csv(OUTPUT_DIR / f"monthly_{name}.csv", index=False)

        print("  exact window metrics ...")
        row = {"scope": name, **exact_window_metrics(scope)}

        print("  sample metrics ...")
        row.update(sample_metrics(sample(scope)))

        row["seconds_spent"] = round(time.time() - started, 1)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("scope")
    summary.to_csv(OUTPUT_DIR / "summary.csv")

    pd.set_option("display.width", 200)
    print("\n\n========== SUMMARY (candidates as columns) ==========")
    print(summary.T.to_string())

    print("\n\n========== OPEN % BY MONTH ==========")
    monthly = {name: pd.read_csv(OUTPUT_DIR / f"monthly_{name}.csv").set_index("month")["pct_open"]
               for name in CANDIDATES}
    print(pd.DataFrame(monthly).to_string())


if __name__ == "__main__":
    main()