"""
Fetches all required data from SSB (Statistics Norway) API.
POST to https://data.ssb.no/api/v0/no/table/{tableId}
Response: JSON-stat2 format. No auth required.
"""
import requests
import json
import time
import sys
from pathlib import Path

SSB_API_BASE = "https://data.ssb.no/api/v0/no/table"
DATA_DIR = Path(__file__).parent
RATE_LIMIT_DELAY = 2.5  # seconds between requests


def generate_month_codes(start_year, start_month, end_year, end_month):
    codes = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        codes.append(f"{y}M{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return codes


def generate_quarter_codes(start_year, start_q, end_year, end_q):
    codes = []
    y, q = start_year, start_q
    while (y, q) <= (end_year, end_q):
        codes.append(f"{y}K{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return codes


def generate_year_codes(start, end):
    return [str(y) for y in range(start, end + 1)]


def fetch_table(table_id, query_body, output_path, max_retries=3):
    url = f"{SSB_API_BASE}/{table_id}"
    print(f"  Fetching table {table_id} -> {output_path.name}...", end=" ", flush=True)

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=query_body, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if "value" not in data:
                    print(f"WARNING: no 'value' key in response")
                    return False
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                n_values = len(data.get("value", []))
                print(f"OK ({n_values} values)")
                return True
            elif resp.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif resp.status_code == 403:
                print(f"FORBIDDEN (table may not exist or query invalid)")
                print(f"  Response: {resp.text[:300]}")
                return False
            else:
                print(f"HTTP {resp.status_code}")
                print(f"  Response: {resp.text[:300]}")
                if attempt < max_retries - 1:
                    time.sleep(3)
        except requests.exceptions.Timeout:
            print(f"timeout (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(3)
        except requests.exceptions.ConnectionError as e:
            print(f"connection error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    print(f"FAILED after {max_retries} attempts")
    return False


# ── Query builders ──────────────────────────────────────────────

def build_cpi_query():
    """Table 03013: Consumer Price Index, monthly 2005-2025"""
    months = generate_month_codes(2005, 1, 2025, 12)
    return {
        "query": [
            {"code": "Konsumgrp", "selection": {"filter": "item", "values": ["TOTAL"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["KpiIndMnd", "Tolvmanedersendring"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": months}},
        ],
        "response": {"format": "json-stat2"},
    }


def build_policy_rate_query():
    """Table 10701: Norges Bank policy rate, monthly 2013M12-2026M01"""
    months = generate_month_codes(2013, 12, 2026, 1)
    return {
        "query": [
            {"code": "RenterNbNibor", "selection": {"filter": "item", "values": ["02"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Renter"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": months}},
        ],
        "response": {"format": "json-stat2"},
    }


def build_population_query():
    """Table 01222: Quarterly population change by county, 2005-2024"""
    quarters = generate_quarter_codes(2005, 1, 2024, 4)
    # Use plain numeric codes (no F- prefix) + historical merged codes
    county_codes = [
        "31", "32", "03", "34", "33", "39", "40",
        "42", "11", "46", "15", "50", "18", "55", "56",
        "30", "38", "54",  # Merged (Viken, Vestfold og Telemark, Troms og Finnmark)
        "01", "02", "04", "05", "06", "07", "08", "09", "10",
        "12", "14", "16", "17", "19", "20",  # Historical pre-2020 codes
    ]
    return {
        "query": [
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Folketilvekst10"]}},
            {"code": "Region", "selection": {"filter": "item", "values": county_codes}},
            {"code": "Tid", "selection": {"filter": "item", "values": quarters}},
        ],
        "response": {"format": "json-stat2"},
    }


def build_price_index_query():
    """Table 07221: House price index by region, quarterly 2005-2024"""
    quarters = generate_quarter_codes(2005, 1, 2024, 4)
    region_codes = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011"]
    return {
        "query": [
            {"code": "Boligtype", "selection": {"filter": "item", "values": ["00"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Boligindeks"]}},
            {"code": "Region", "selection": {"filter": "item", "values": region_codes}},
            {"code": "Tid", "selection": {"filter": "item", "values": quarters}},
        ],
        "response": {"format": "json-stat2"},
    }


def build_sales_volume_query():
    """Table 10187: Property sales volume by county, quarterly 2008-2024"""
    quarters = generate_quarter_codes(2008, 1, 2024, 4)
    # Include both current and merged county codes
    region_codes = [
        "0",  # National
        "31", "32", "03", "34", "33", "39", "40",
        "42", "11", "46", "15", "50", "18", "55", "56",
        "30", "38", "54",  # Merged codes (Viken, Vestfold og Telemark, Troms og Finnmark)
    ]
    return {
        "query": [
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Omsetninger"]}},
            {"code": "Region", "selection": {"filter": "item", "values": region_codes}},
            {"code": "Tid", "selection": {"filter": "item", "values": quarters}},
        ],
        "response": {"format": "json-stat2"},
    }


def build_unemployment_query():
    """Table 13760: Unemployment rate, monthly 2006-2025, national, seasonally adjusted"""
    months = generate_month_codes(2006, 1, 2026, 1)
    return {
        "query": [
            {"code": "Kjonn", "selection": {"filter": "item", "values": ["0"]}},
            {"code": "Alder", "selection": {"filter": "item", "values": ["15-74"]}},
            {"code": "Justering", "selection": {"filter": "item", "values": ["S"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["ArbledProsArbstyrk"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": months}},
        ],
        "response": {"format": "json-stat2"},
    }


def build_building_starts_query():
    """Table 03723: Housing starts by county, monthly 2005-2025"""
    months = generate_month_codes(2005, 1, 2026, 1)
    county_codes = [
        "31", "32", "03", "34", "33", "39", "40",
        "42", "11", "46", "15", "50", "18", "55", "56",
        "30", "38", "54",
        "01", "02", "04", "05", "06", "07", "08", "09", "10",
        "12", "14", "16", "17", "19", "20",  # Historical pre-2020 codes
    ]
    return {
        "query": [
            {"code": "Byggeareal", "selection": {"filter": "item", "values": ["_T"]}},
            {"code": "Region", "selection": {"filter": "item", "values": county_codes}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["BoligIgang"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": months}},
        ],
        "response": {"format": "json-stat2"},
    }


def build_mortgage_rate_query():
    """Table 10748: Mortgage interest rates, monthly 2014-2025"""
    months = generate_month_codes(2014, 1, 2025, 12)
    return {
        "query": [
            {"code": "Utlanstype", "selection": {"filter": "item", "values": ["04"]}},
            {"code": "Sektor", "selection": {"filter": "item", "values": ["04b"]}},
            {"code": "Rentebinding", "selection": {"filter": "item", "values": ["08"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["RenterNyeBolig"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": months}},
        ],
        "response": {"format": "json-stat2"},
    }


def build_gdp_query():
    """Table 09171: GDP volume change, quarterly 2005-2025, national"""
    quarters = generate_quarter_codes(2005, 1, 2025, 4)
    return {
        "query": [
            {"code": "NACE", "selection": {"filter": "item", "values": ["nr23_6"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["BNPB4"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": quarters}},
        ],
        "response": {"format": "json-stat2"},
    }


def build_household_income_query():
    """Table 06944: Median household income after tax, annual 2005-2023, county-level"""
    years = generate_year_codes(2005, 2023)
    county_codes = [
        "31", "32", "03", "34", "33", "39", "40",
        "42", "11", "46", "15", "50", "18", "55", "56",
        "30", "38", "54",
    ]
    return {
        "query": [
            {"code": "Region", "selection": {"filter": "item", "values": county_codes}},
            {"code": "HusholdType", "selection": {"filter": "item", "values": ["0000"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["InntSkatt"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": years}},
        ],
        "response": {"format": "json-stat2"},
    }


# ── Main ────────────────────────────────────────────────────────

def main():
    queries = [
        ("03013", build_cpi_query(), "kpi.json"),
        ("10701", build_policy_rate_query(), "policy_rate.json"),
        ("01222", build_population_query(), "population_change.json"),
        ("07221", build_price_index_query(), "property_price_index.json"),
        ("10187", build_sales_volume_query(), "revenue_properties.json"),
        ("13760", build_unemployment_query(), "unemployment.json"),
        ("03723", build_building_starts_query(), "building_starts.json"),
        ("10748", build_mortgage_rate_query(), "mortgage_rate.json"),
        ("09171", build_gdp_query(), "gdp.json"),
        ("06944", build_household_income_query(), "household_income.json"),
    ]

    print(f"Fetching {len(queries)} tables from SSB API...")
    success_count = 0
    failed = []

    for table_id, query_body, filename in queries:
        output_path = DATA_DIR / filename
        ok = fetch_table(table_id, query_body, output_path)
        if ok:
            success_count += 1
        else:
            failed.append(filename)
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone: {success_count}/{len(queries)} tables fetched successfully.")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
