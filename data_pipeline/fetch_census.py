"""
fetch_census.py — Download ACS Census data for Chicago Census tracts.

Data source: US Census Bureau American Community Survey (ACS) 5-Year Estimates.
API: census.gov/data API (free, requires key from census.gov/developers).

Variables fetched (all tract-level aggregates — NO individual data):
- Median household income (B19013_001E)
- Total population (B01003_001E)
- Housing unit count (B25001_001E)
- % households without a vehicle (B08201_002E / B08201_001E)
- % population below poverty line (B17001_002E / B17001_001E)
- % with bachelor's degree or higher (B15003_022E+ / B15003_001E)

Ethics:
- All data is publicly published at Census tract level (min ~4,000 people)
- No individual or household-level data is fetched or stored
- Missing tracts documented in bias audit

Usage:
    python fetch_census.py --api-key $CENSUS_API_KEY \
        --year 2022 --output data/raw/census/
"""

import os
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# ── ACS variable definitions ────────────────────────────────────────────────────

ACS_VARIABLES = {
    # Income
    "B19013_001E": "median_household_income",
    # Population
    "B01003_001E": "total_population",
    # Housing
    "B25001_001E": "total_housing_units",
    # Vehicle access (proxy for transit dependence)
    "B08201_001E": "households_total_vehicles",
    "B08201_002E": "households_no_vehicle",
    # Poverty
    "B17001_001E": "poverty_universe",
    "B17001_002E": "poverty_below_line",
    # Education
    "B15003_001E": "education_universe",
    "B15003_022E": "bachelors_degree",
    "B15003_023E": "masters_degree",
    "B15003_025E": "doctorate_degree",
}

# Illinois FIPS = 17, Cook County FIPS = 031
STATE_FIPS = "17"
COUNTY_FIPS = "031"
ACS_BASE_URL = "https://api.census.gov/data"


# ── Census API fetcher ──────────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=30))
def _fetch_acs(
    variables: list[str],
    year: int,
    api_key: str,
    state: str = STATE_FIPS,
    county: str = COUNTY_FIPS,
) -> pd.DataFrame:
    """Fetch ACS 5-year estimates for all Cook County Census tracts.

    Retrieves data from the US Census Bureau API for specified variables.
    All data is publicly available at Census tract level (minimum ~4,000 people).
    Retries automatically up to 4 times on network failures.

    Args:
        variables: List of ACS variable codes (e.g., ['B19013_001E', 'B01003_001E'])
        year: ACS vintage year (e.g., 2022 for 2018-2022 estimates)
        api_key: Census Bureau API key (free from census.gov/developers)
        state: State FIPS code (default: '17' for Illinois)
        county: County FIPS code (default: '031' for Cook County)

    Returns:
        DataFrame with one row per Census tract. Columns include variable codes
        and geographic identifiers (state, county, tract). Column 'NAME' has
        human-readable tract descriptions.

    Raises:
        requests.HTTPError: If API request fails after retries
        ValueError: If API returns error status
    """
    var_string = "NAME," + ",".join(variables)
    url = f"{ACS_BASE_URL}/{year}/acs/acs5"
    params = {
        "get": var_string,
        "for": "tract:*",
        "in": f"state:{state} county:{county}",
        "key": api_key,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    columns = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=columns)
    logger.info(f"ACS {year}: fetched {len(df)} tracts, {len(columns)} columns.")
    return df


def _clean_acs(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Clean and validate ACS data, computing derived metrics.

    Performs the following transformations:
    - Creates standardized 11-digit tract_id (FIPS: state + county + tract)
    - Converts string columns to numeric, replacing Census sentinel values with NaN
    - Computes derived rates (poverty_rate, pct_no_vehicle, pct_college_plus)
    - Handles division by zero safely

    Sentinel values: Census Bureau uses -666666666 to indicate missing/not available data.
    These are converted to NaN for safe downstream processing.

    Args:
        df: Raw DataFrame from Census API with string-valued columns
        year: ACS vintage year (stored in acs_year column for reproducibility)

    Returns:
        Cleaned DataFrame with:
        - tract_id: 11-digit FIPS code (e.g., '17031001100')
        - All numeric variables converted to float64
        - Computed rates: poverty_rate, pct_no_vehicle, pct_college_plus (range [0,1])
        - acs_year: Year of estimate
        - NAME: Tract description from Census

    Notes:
        - Missing values are preserved as NaN (not imputed)
        - Rates are NaN if denominator is zero or missing
        - For education: includes bachelor's + master's + doctorate degrees
    """
    df = df.copy()

    # Build 11-digit FIPS tract ID (standard GeoID format)
    df["tract_id"] = df["state"] + df["county"] + df["tract"]

    # Rename raw API columns
    rename_map = {k: v for k, v in ACS_VARIABLES.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Convert to numeric; replace Census sentinel with NaN
    numeric_cols = list(ACS_VARIABLES.values())
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col] > -999999, other=float("nan"))

    # Derived rates (avoid division by zero)
    def safe_rate(numerator_col: str, denominator_col: str, new_col: str) -> None:
        if numerator_col in df.columns and denominator_col in df.columns:
            denom = df[denominator_col].replace(0, float("nan"))
            df[new_col] = df[numerator_col] / denom

    safe_rate("households_no_vehicle", "households_total_vehicles", "pct_no_vehicle")
    safe_rate("poverty_below_line", "poverty_universe", "poverty_rate")

    # Education: bachelor's + master's + doctorate
    edu_cols = ["bachelors_degree", "masters_degree", "doctorate_degree"]
    present_edu = [c for c in edu_cols if c in df.columns]
    if present_edu and "education_universe" in df.columns:
        df["college_plus_count"] = df[present_edu].sum(axis=1, min_count=1)
        safe_rate("college_plus_count", "education_universe", "pct_college_plus")

    df["acs_year"] = year

    # Final column selection
    keep_cols = [
        "tract_id",
        "NAME",
        "acs_year",
        "median_household_income",
        "total_population",
        "total_housing_units",
        "pct_no_vehicle",
        "poverty_rate",
        "pct_college_plus",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


def _audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and log missingness per column.

    Returns audit DataFrame for inclusion in bias report.
    """
    total = len(df)
    audit_rows = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct_missing = 100 * n_missing / total
        audit_rows.append(
            {
                "column": col,
                "n_missing": n_missing,
                "pct_missing": round(pct_missing, 2),
                "bias_flag": (
                    "HIGH" if pct_missing > 10 else ("MODERATE" if pct_missing > 5 else "OK")
                ),
            }
        )
        if pct_missing > 10:
            logger.warning(f"  {col}: {pct_missing:.1f}% missing — flag for bias report")
        elif pct_missing > 0:
            logger.info(f"  {col}: {pct_missing:.1f}% missing")

    return pd.DataFrame(audit_rows)


def fetch_chicago_tracts_shapefile(
    output_dir: Path,
    year: int = 2022,
) -> gpd.GeoDataFrame:
    """
    Download Census tract geometries for Cook County from Census Bureau TIGER files.

    These are the actual polygon boundaries used to join all data layers.
    Source: US Census Bureau TIGER/Line Shapefiles (public domain).
    """
    tiger_url = (
        f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/" f"tl_{year}_{STATE_FIPS}_tract.zip"
    )
    shapefile_path = output_dir / f"cook_county_tracts_{year}.geojson"

    if shapefile_path.exists():
        logger.info(f"Tract shapefile already exists: {shapefile_path}")
        return gpd.read_file(shapefile_path)

    logger.info(f"Downloading TIGER tract geometries for Illinois ({year})...")
    import io
    import zipfile

    import requests as req

    resp = req.get(tiger_url, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(output_dir / "tiger_tmp")

    # Read shapefile from extracted dir
    shp_files = list((output_dir / "tiger_tmp").glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No shapefile found in TIGER download.")

    all_tracts = gpd.read_file(shp_files[0])
    # Filter to Cook County
    cook_tracts = all_tracts[all_tracts["COUNTYFP"] == COUNTY_FIPS].copy()
    cook_tracts["tract_id"] = cook_tracts["GEOID"]  # 11-digit FIPS
    cook_tracts = cook_tracts.to_crs(epsg=4326)

    # Keep only essential columns
    keep = ["tract_id", "NAMELSAD", "ALAND", "AWATER", "geometry"]
    keep = [c for c in keep if c in cook_tracts.columns]
    cook_tracts = cook_tracts[keep]
    cook_tracts.to_file(shapefile_path, driver="GeoJSON")

    # Cleanup temp files
    import shutil

    shutil.rmtree(output_dir / "tiger_tmp", ignore_errors=True)

    logger.info(f"Cook County tracts: {len(cook_tracts)} → {shapefile_path}")
    return cook_tracts


# ── Main ────────────────────────────────────────────────────────────────────────


def fetch_all(
    api_key: str,
    year: int = 2022,
    output_dir: str | Path = "data/raw/census/",
) -> dict[str, Path]:
    """
    Fetch ACS income/demographics + tract geometries for Cook County.

    Returns dict of {name: path}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. Tract geometries (for spatial joins later)
    tracts_gdf = fetch_chicago_tracts_shapefile(output_dir, year=year)
    tracts_path = output_dir / f"cook_county_tracts_{year}.geojson"
    paths["tracts_geometry"] = tracts_path

    # Also save centroid file for Street View sampling
    centroids = tracts_gdf.copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids_path = output_dir / "tract_centroids.geojson"
    centroids.to_file(centroids_path, driver="GeoJSON")
    # Copy to processed for use by fetch_streetview.py
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    centroids.to_file(processed_dir / "tract_centroids.geojson", driver="GeoJSON")
    paths["tract_centroids"] = centroids_path
    logger.info(f"Tract centroids saved → {centroids_path}")

    # 2. ACS income + socioeconomic data
    acs_path = output_dir / f"acs_{year}_cook_county.csv"
    if acs_path.exists():
        logger.info(f"ACS data already fetched: {acs_path}")
        acs_df = pd.read_csv(acs_path)
    else:
        raw_df = _fetch_acs(
            variables=list(ACS_VARIABLES.keys()),
            year=year,
            api_key=api_key,
        )
        acs_df = _clean_acs(raw_df, year=year)
        acs_df.to_csv(acs_path, index=False)
        logger.info(f"ACS data saved: {len(acs_df)} tracts → {acs_path}")
    paths["acs_data"] = acs_path

    # 3. Missingness audit
    logger.info("Running missingness audit on ACS data:")
    audit_df = _audit_missing(acs_df)
    audit_path = Path("data/audit") / f"census_missing_audit_{year}.csv"
    Path("data/audit").mkdir(exist_ok=True)
    audit_df.to_csv(audit_path, index=False)
    paths["census_audit"] = audit_path

    # Summary
    logger.info(f"\nACS Summary ({year}):")
    logger.info(f"  Tracts: {len(acs_df)}")
    logger.info(
        f"  Median income range: "
        f"${acs_df['median_household_income'].min():,.0f} – "
        f"${acs_df['median_household_income'].max():,.0f}"
    )
    logger.info(
        f"  Poverty rate range: "
        f"{100*acs_df['poverty_rate'].min():.1f}% – "
        f"{100*acs_df['poverty_rate'].max():.1f}%"
    )

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Chicago ACS Census data")
    parser.add_argument(
        "--api-key", required=True, help="Census API key (from census.gov/developers)"
    )
    parser.add_argument("--year", type=int, default=2022, help="ACS 5-year estimate year")
    parser.add_argument("--output", default="data/raw/census/", help="Output directory")
    args = parser.parse_args()

    paths = fetch_all(
        api_key=args.api_key,
        year=args.year,
        output_dir=args.output,
    )
    for name, path in paths.items():
        print(f"  {name:30s} → {path}")
