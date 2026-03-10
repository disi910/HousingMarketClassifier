import pandas as pd
import json
import numpy as np
from pathlib import Path
from itertools import product


class PropertyMarketDataParser:
    # County code -> canonical county name (current + historical + merged)
    COUNTY_CODE_MAP = {
        # Current counties (2024+)
        "31": "Østfold", "32": "Akershus", "03": "Oslo",
        "34": "Innlandet", "33": "Buskerud", "39": "Vestfold",
        "40": "Telemark", "42": "Agder", "11": "Rogaland",
        "46": "Vestland", "15": "Møre og Romsdal", "50": "Trøndelag",
        "18": "Nordland", "55": "Troms", "56": "Finnmark",
        # Merged 2020-2023
        "30": "Viken", "38": "Vestfold og Telemark",
        "54": "Troms og Finnmark",
        # Historical pre-2020
        "01": "Østfold", "02": "Akershus", "04": "Hedmark",
        "05": "Oppland", "06": "Buskerud", "07": "Vestfold",
        "08": "Telemark", "09": "Aust-Agder", "10": "Vest-Agder",
        "12": "Hordaland", "14": "Sogn og Fjordane",
        "16": "Sør-Trøndelag", "17": "Nord-Trøndelag",
        "19": "Troms", "20": "Finnmark",
    }

    # Map historical county names to the modern canonical name
    HISTORICAL_TO_MODERN = {
        "Hedmark": "Innlandet", "Oppland": "Innlandet",
        "Aust-Agder": "Agder", "Vest-Agder": "Agder",
        "Hordaland": "Vestland", "Sogn og Fjordane": "Vestland",
        "Sør-Trøndelag": "Trøndelag", "Nord-Trøndelag": "Trøndelag",
        # Merged counties split back
        "Viken": None,  # Split into Østfold, Akershus, Buskerud
        "Vestfold og Telemark": None,  # Split into Vestfold, Telemark
        "Troms og Finnmark": None,  # Split into Troms, Finnmark
    }

    # Merged county -> constituent modern counties (for distributing data)
    MERGED_SPLITS = {
        "Viken": ["Østfold", "Akershus", "Buskerud"],
        "Vestfold og Telemark": ["Vestfold", "Telemark"],
        "Troms og Finnmark": ["Troms", "Finnmark"],
    }

    # 15 canonical counties we produce output for
    CANONICAL_COUNTIES = [
        "Østfold", "Akershus", "Oslo", "Innlandet", "Buskerud",
        "Vestfold", "Telemark", "Agder", "Rogaland", "Vestland",
        "Møre og Romsdal", "Trøndelag", "Nordland", "Troms", "Finnmark",
    ]

    # County -> price index region (table 07221)
    COUNTY_TO_PRICE_REGION = {
        "Oslo": "Oslo med Bærum", "Akershus": "Akershus uten Bærum",
        "Rogaland": "Stavanger", "Vestland": "Bergen",
        "Trøndelag": "Trondheim", "Innlandet": "Innlandet",
        "Agder": "Agder og Rogaland uten Stavanger",
        "Nordland": "Nord-Norge", "Troms": "Nord-Norge", "Finnmark": "Nord-Norge",
        "Møre og Romsdal": "Møre og Romsdal og Vestland uten Bergen",
        "Østfold": "Østfold, Buskerud, Vestfold og Telemark",
        "Buskerud": "Østfold, Buskerud, Vestfold og Telemark",
        "Vestfold": "Østfold, Buskerud, Vestfold og Telemark",
        "Telemark": "Østfold, Buskerud, Vestfold og Telemark",
    }

    PRICE_REGION_CODES = {
        "001": "Oslo med Bærum", "002": "Stavanger", "003": "Bergen",
        "004": "Trondheim", "005": "Akershus uten Bærum",
        "006": "Østfold, Buskerud, Vestfold og Telemark",
        "007": "Innlandet",
        "008": "Agder og Rogaland uten Stavanger",
        "009": "Møre og Romsdal og Vestland uten Bergen",
        "010": "Trøndelag uten Trondheim", "011": "Nord-Norge",
    }

    def __init__(self, data_dir=None):
        if data_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            self.data_dir = project_root / "data"
        else:
            self.data_dir = Path(data_dir)
        print(f"Data dir: {self.data_dir}")

    # ── Generic JSON-stat2 parser ───────────────────────────────

    def parse_jsonstat2(self, filename):
        """Parse any SSB JSON-stat2 response into a flat DataFrame."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"  WARNING: {filename} not found, skipping")
            return pd.DataFrame()

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        dim_ids = data["id"]
        sizes = data["size"]
        values = data["value"]

        # Build ordered code lists for each dimension
        dim_codes = []
        for dim_id in dim_ids:
            cat = data["dimension"][dim_id]["category"]
            idx_map = cat["index"]
            # Sort codes by their index position
            codes = sorted(idx_map.keys(), key=lambda c: idx_map[c])
            dim_codes.append(codes)

        # Build all index combos (cartesian product)
        rows = list(product(*dim_codes))
        df = pd.DataFrame(rows, columns=dim_ids)
        df["value"] = values[:len(rows)]
        return df

    # ── Transform helpers ───────────────────────────────────────

    @staticmethod
    def _monthly_to_quarterly(df, value_col="value"):
        """Convert monthly Tid (YYYYMXX) to quarterly averages."""
        df = df.copy()
        df["year"] = df["Tid"].str[:4].astype(int)
        df["month"] = df["Tid"].str[5:7].astype(int)
        df["quarter_num"] = ((df["month"] - 1) // 3) + 1
        df["quarter"] = df["year"].astype(str) + "K" + df["quarter_num"].astype(str)
        return df

    def _resolve_county(self, code):
        """Resolve a region code to a canonical modern county name."""
        raw = self.COUNTY_CODE_MAP.get(str(code), str(code))
        # Clean SSB label suffixes
        for suffix in [" - Oslove", " - Nordlánnda", " - Romsa - Tromssa",
                       " - Finnmárku - Finmarkku", " - Trööndelage",
                       " - Romsa ja Finnmárku"]:
            raw = raw.replace(suffix, "")
        raw = raw.split(" (")[0].strip()
        return self.HISTORICAL_TO_MODERN.get(raw, raw) or raw

    def _distribute_merged(self, county_data):
        """Distribute merged county data across constituent counties using equal split."""
        result = {}
        for county, qdata in county_data.items():
            if county in self.MERGED_SPLITS:
                parts = self.MERGED_SPLITS[county]
                for part in parts:
                    if part not in result:
                        result[part] = {}
                    for q, val in qdata.items():
                        if q not in result[part]:
                            result[part][q] = val / len(parts) if val is not None else None
            else:
                modern = self.HISTORICAL_TO_MODERN.get(county, county) or county
                if modern not in result:
                    result[modern] = {}
                for q, val in qdata.items():
                    # Sum if multiple historical counties map to same modern one
                    result[modern][q] = result[modern].get(q, 0) + (val if val is not None else 0)
        return result

    # ── Per-dataset transforms ──────────────────────────────────

    def transform_cpi(self):
        """CPI monthly -> quarterly averages (national)."""
        df = self.parse_jsonstat2("kpi.json")
        if df.empty:
            return {}
        # Filter to KpiIndMnd (index values)
        if "ContentsCode" in df.columns:
            df = df[df["ContentsCode"] == "KpiIndMnd"]
        df = self._monthly_to_quarterly(df)
        return df.groupby("quarter")["value"].mean().to_dict()

    def transform_policy_rate(self):
        """Policy rate monthly -> quarterly averages (national)."""
        df = self.parse_jsonstat2("policy_rate.json")
        if df.empty:
            return {}
        df = self._monthly_to_quarterly(df)
        return df.groupby("quarter")["value"].mean().to_dict()

    def transform_population(self):
        """Population change quarterly by county -> {county: {quarter: val}}."""
        df = self.parse_jsonstat2("population_change.json")
        if df.empty:
            return {}
        county_data = {}
        for _, row in df.iterrows():
            county_raw = self.COUNTY_CODE_MAP.get(row["Region"], row["Region"])
            county = self.HISTORICAL_TO_MODERN.get(county_raw, county_raw) or county_raw
            if county not in county_data:
                county_data[county] = {}
            q = row["Tid"]
            val = row["value"]
            # Sum if multiple codes map to same modern county
            county_data[county][q] = county_data[county].get(q, 0) + (val if val is not None else 0)

        # Handle merged counties
        for merged, parts in self.MERGED_SPLITS.items():
            if merged in county_data:
                for part in parts:
                    if part not in county_data:
                        county_data[part] = {}
                    for q, val in county_data[merged].items():
                        if q not in county_data[part]:
                            county_data[part][q] = val / len(parts) if val else 0
                del county_data[merged]

        return county_data

    def transform_price_index(self):
        """Price index quarterly by region -> {region_label: {quarter: val}}."""
        df = self.parse_jsonstat2("property_price_index.json")
        if df.empty:
            return {}
        result = {}
        for _, row in df.iterrows():
            region_label = self.PRICE_REGION_CODES.get(row["Region"], row["Region"])
            if region_label not in result:
                result[region_label] = {}
            result[region_label][row["Tid"]] = row["value"]
        return result

    def transform_revenue(self):
        """Sales volume quarterly by county -> {county: {quarter: val}}."""
        # Try single file first (new format), fall back to old split files
        df = self.parse_jsonstat2("revenue_properties.json")
        if df.empty:
            df_old = self.parse_jsonstat2("revenue_properties_2020_2023.json")
            df_new = self.parse_jsonstat2("revenue_properties_2024.json")
            df = pd.concat([df_old, df_new], ignore_index=True)
        if df.empty:
            return {}

        county_data = {}
        for _, row in df.iterrows():
            code = row["Region"]
            county_raw = self.COUNTY_CODE_MAP.get(code, code)
            # Clean labels from SSB
            if isinstance(county_raw, str):
                county_raw = county_raw.split(" (")[0].split(" - ")[0].strip()
            county = self.HISTORICAL_TO_MODERN.get(county_raw, county_raw) or county_raw
            if county == "0" or code == "0":
                continue  # Skip national total
            if county not in county_data:
                county_data[county] = {}
            q = row["Tid"]
            val = row["value"]
            county_data[county][q] = county_data[county].get(q, 0) + (val if val is not None else 0)

        # Distribute merged counties
        for merged, parts in self.MERGED_SPLITS.items():
            if merged in county_data:
                for part in parts:
                    if part not in county_data:
                        county_data[part] = {}
                    for q, val in county_data[merged].items():
                        if q not in county_data[part]:
                            county_data[part][q] = val / len(parts) if val else 0
                del county_data[merged]

        return county_data

    def transform_unemployment(self):
        """Unemployment rate monthly -> quarterly averages (national)."""
        df = self.parse_jsonstat2("unemployment.json")
        if df.empty:
            return {}
        df = self._monthly_to_quarterly(df)
        return df.groupby("quarter")["value"].mean().to_dict()

    def transform_building_starts(self):
        """Building starts monthly by county -> quarterly sums by county."""
        df = self.parse_jsonstat2("building_starts.json")
        if df.empty:
            return {}
        df = self._monthly_to_quarterly(df)

        county_data = {}
        for (region, quarter), group in df.groupby(["Region", "quarter"]):
            county_raw = self.COUNTY_CODE_MAP.get(region, region)
            county = self.HISTORICAL_TO_MODERN.get(county_raw, county_raw) or county_raw
            if county not in county_data:
                county_data[county] = {}
            total = group["value"].sum()
            county_data[county][quarter] = county_data[county].get(quarter, 0) + total

        # Distribute merged counties
        for merged, parts in self.MERGED_SPLITS.items():
            if merged in county_data:
                for part in parts:
                    if part not in county_data:
                        county_data[part] = {}
                    for q, val in county_data[merged].items():
                        if q not in county_data[part]:
                            county_data[part][q] = val / len(parts) if val else 0
                del county_data[merged]

        return county_data

    def transform_mortgage_rate(self):
        """Mortgage rate monthly -> quarterly averages (national)."""
        df = self.parse_jsonstat2("mortgage_rate.json")
        if df.empty:
            return {}
        df = self._monthly_to_quarterly(df)
        return df.groupby("quarter")["value"].mean().to_dict()

    def transform_gdp(self):
        """GDP volume change quarterly (national)."""
        df = self.parse_jsonstat2("gdp.json")
        if df.empty:
            return {}
        return dict(zip(df["Tid"], df["value"]))

    def transform_household_income(self):
        """Household income annual by county -> broadcast to all 4 quarters."""
        df = self.parse_jsonstat2("household_income.json")
        if df.empty:
            return {}

        county_data = {}
        for _, row in df.iterrows():
            code = row["Region"]
            county_raw = self.COUNTY_CODE_MAP.get(code, code)
            county = self.HISTORICAL_TO_MODERN.get(county_raw, county_raw) or county_raw
            if county not in county_data:
                county_data[county] = {}
            year = row["Tid"]
            val = row["value"]
            for q in range(1, 5):
                county_data[county][f"{year}K{q}"] = val

        # Distribute merged counties
        for merged, parts in self.MERGED_SPLITS.items():
            if merged in county_data:
                for part in parts:
                    if part not in county_data:
                        county_data[part] = {}
                    for q, val in county_data[merged].items():
                        if q not in county_data[part]:
                            county_data[part][q] = val
                del county_data[merged]

        return county_data

    # ── Unified dataset ─────────────────────────────────────────

    def create_unified_dataset(self):
        """Create unified dataset with all features."""
        print("=== Parsing all data sources ===")

        cpi_data = self.transform_cpi()
        policy_rate = self.transform_policy_rate()
        population = self.transform_population()
        price_index = self.transform_price_index()
        revenue = self.transform_revenue()
        unemployment = self.transform_unemployment()
        building_starts = self.transform_building_starts()
        mortgage_rate = self.transform_mortgage_rate()
        gdp_change = self.transform_gdp()
        household_income = self.transform_household_income()

        print("=== Creating unified dataset ===")

        # Derive quarter range from price index data
        all_quarters = sorted(set(
            q for region_data in price_index.values() for q in region_data.keys()
        ))
        if not all_quarters:
            all_quarters = [f"{y}K{q}" for y in range(2005, 2025) for q in range(1, 5)]

        data_rows = []
        for county in self.CANONICAL_COUNTIES:
            price_region = self.COUNTY_TO_PRICE_REGION.get(county)

            for quarter in all_quarters:
                year = int(quarter[:4])
                q_num = int(quarter[-1])

                row = {
                    "region": county,
                    "quarter": quarter,
                    "year": year,
                    "quarter_num": q_num,
                    "cpi": cpi_data.get(quarter),
                    "policy_rate": policy_rate.get(quarter),
                    "population_change": population.get(county, {}).get(quarter),
                    "sales_volume": revenue.get(county, {}).get(quarter),
                    "unemployment_rate": unemployment.get(quarter),
                    "building_starts": building_starts.get(county, {}).get(quarter),
                    "mortgage_rate": mortgage_rate.get(quarter),
                    "gdp_change": gdp_change.get(quarter),
                    "household_income": household_income.get(county, {}).get(quarter),
                }

                # Price index lookup
                if price_region and price_region in price_index:
                    row["price_index"] = price_index[price_region].get(quarter)
                else:
                    row["price_index"] = None

                data_rows.append(row)

        df = pd.DataFrame(data_rows)
        # Only require price_index (target variable dependency)
        df = df.dropna(subset=["price_index"])

        return df

    def save_processed_data(self, df, output_file="processed_data.csv"):
        output_dir = Path(__file__).resolve().parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_file
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved processed data to: {output_path.absolute()}")
        return output_path


if __name__ == "__main__":
    parser = PropertyMarketDataParser()
    df = parser.create_unified_dataset()
    parser.save_processed_data(df)

    print(f"\nData: {len(df)} rows, {df['region'].nunique()} regions, "
          f"{df['year'].min()}-{df['year'].max()}, {len(df.columns)} columns")
