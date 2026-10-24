"""Compare candidate scopes before choosing the scope criterion."""
import requests

BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
META_URL = "https://data.cityofnewyork.us/api/views/erm2-nwe9.json"
WINDOW = "created_date >= '2024-01-01T00:00:00' AND created_date < '2025-07-01T00:00:00'"
CANDIDATES = ["DOT", "DSNY", "HPD"]

def ask(params):
    r = requests.get(BASE_URL, params=params, timeout=120)
    r.raise_for_status()
    return r.json()

# 0) Official column list from metadata (not from a data sample)
meta = requests.get(META_URL, timeout=60).json()
print("official columns:")
for col in meta["columns"]:
    print(f"  {col['fieldName']:35} {col['dataTypeName']}")

for agency in CANDIDATES:
    where = f"{WINDOW} AND agency='{agency}'"
    print(f"\n===== {agency} =====")

    # 1) Status distribution and how many are still not closed
    print("by status:", ask({"$select": "status, count(*) AS n", "$where": where,
                             "$group": "status", "$order": "n DESC"}))
    print("closed_date is null:", ask({"$select": "count(*)",
                                       "$where": where + " AND closed_date IS NULL"}))

    # 2) Number of distinct complaint types and the top ones
    types = ask({"$select": "complaint_type, count(*) AS n", "$where": where,
                 "$group": "complaint_type", "$order": "n DESC", "$limit": 50})
    print("distinct complaint types (up to 50):", len(types))
    for row in types[:10]:
        print("  ", row)