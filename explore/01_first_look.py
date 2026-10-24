"""Exploratory requests to understand the NYC 311 API. Not part of the final package."""
import json
import requests

BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

def ask(params):
    response = requests.get(BASE_URL, params=params, timeout=60)
    print("status:", response.status_code, "| url:", response.url)
    response.raise_for_status()
    return response.json()

# 1) A few raw records: which fields exist?
sample = ask({"$limit": 5})
all_keys = set()
for record in sample:
    all_keys.update(record.keys())
print("number of fields per record:", [len(r) for r in sample])
print("all fields seen:", sorted(all_keys))
print(json.dumps(sample[0], indent=2))



START = "2024-01-01T00:00:00"
END = "2025-07-01T00:00:00"   # 18 months, end is exclusive
window = f"created_date >= '{START}' AND created_date < '{END}'"

# 2) Total records in the 18-month window
total = ask({"$select": "count(*)", "$where": window})
print("total in window:", total)

# 3) Records per agency in that window
by_agency = ask({
    "$select": "agency, count(*) AS n",
    "$where": window,
    "$group": "agency",
    "$order": "n DESC",
})
for row in by_agency:
    print(row)