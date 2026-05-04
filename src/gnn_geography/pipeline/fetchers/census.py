"""Census ACS data fetcher."""

import io
import os
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ...config import Config
from .base import BaseFetcher

# Illinois FIPS = 17, Cook County FIPS = 031
STATE_FIPS = "17"
COUNTY_FIPS = "031"
ACS_BASE_URL = "https://api.census.gov/data"


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
    return df


def _clean_acs(df: pd.DataFrame, year: int, acs_variables: dict) -> pd.DataFrame:
    """Clean and validate ACS data, computing derived metrics."""
    df = df.copy()

    # Build 11-digit FIPS tract ID
    df["tract_id"] = df["state"] + df["county"] + df["tract"]

    # Rename raw API columns
    rename_map = {k: v for k, v in acs_variables.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Convert to numeric; replace Census sentinel with NaN
    numeric_cols = list(acs_variables.values())
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
    """Compute and log missingness per column."""
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
            self.logger.warning(f"  {col}: {pct_missing:.1f}% missing")
        elif pct_missing > 0:
            self.logger.info(f"  {col}: {pct_missing:.1f}% missing")

    return pd.DataFrame(audit_rows)


def fetch_chicago_tracts_shapefile(
    output_dir: Path,
    year: int = 2022,
) -> gpd.GeoDataFrame:
    """Download Census tract geometries for Cook County from TIGER files."""
    tiger_url = (
        f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
        f"tl_{year}_{STATE_FIPS}_tract.zip"
    )
    shapefile_path = output_dir / f"cook_county_tracts_{year}.geojson"

    if shapefile_path.exists():
        return gpd.read_file(shapefile_path)

    resp = requests.get(tiger_url, timeout=120)
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
    cook_tracts["tract_id"] = cook_tracts["GEOID"]
    cook_tracts = cook_tracts.to_crs(epsg=4326)

    # Keep only essential columns
    keep = ["tract_id", "NAMELSAD", "ALAND", "AWATER", "geometry"]
    keep = [c for c in keep if c in cook_tracts.columns]
    cook_tracts = cook_tracts[keep]
    cook_tracts.to_file(shapefile_path, driver="GeoJSON")

    # Cleanup temp files
    shutil.rmtree(output_dir / "tiger_tmp", ignore_errors=True)

    return cook_tracts


class CensusFetcher(BaseFetcher):
    """Fetcher for US Census ACS data."""

    def __init__(self, config: Config):
        """Initialize Census fetcher.

        Args:
            config: Configuration instance with Census settings
        """
        super().__init__(config, "census")

    def fetch(self) -> dict[str, Path]:
        """Fetch ACS income/demographics + tract geometries for Cook County.

        Returns:
            Dictionary mapping output names to file paths
        """
        census_config = self.config.census
        output_dir = Path(self.config.paths.census_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # 1. Tract geometries
        tracts_gdf = fetch_chicago_tracts_shapefile(output_dir, year=census_config.acs_year)
        tracts_path = output_dir / f"cook_county_tracts_{census_config.acs_year}.geojson"
        paths["tracts_geometry"] = tracts_path

        # Also save centroid file for Street View sampling
        centroids = tracts_gdf.copy()
        centroids["geometry"] = centroids.geometry.centroid
        centroids_path = output_dir / "tract_centroids.geojson"
        centroids.to_file(centroids_path, driver="GeoJSON")
        # Copy to processed
        processed_dir = Path(self.config.paths.processed)
        processed_dir.mkdir(exist_ok=True)
        centroids.to_file(processed_dir / "tract_centroids.geojson", driver="GeoJSON")
        paths["tract_centroids"] = centroids_path
        self.logger.info(f"Tract centroids saved → {centroids_path}")

        # 2. ACS income + socioeconomic data
        acs_path = output_dir / f"acs_{census_config.acs_year}_cook_county.csv"
        if acs_path.exists():
            self.logger.info(f"ACS data already fetched: {acs_path}")
            acs_df = pd.read_csv(acs_path)
        else:
            raw_df = _fetch_acs(
                variables=list(census_config.variables.values()),
                year=census_config.acs_year,
                api_key=os.environ.get("CENSUS_API_KEY", ""),
            )
            acs_df = _clean_acs(raw_df, year=census_config.acs_year,
                               acs_variables=census_config.variables)
            acs_df.to_csv(acs_path, index=False)
            self.logger.info(f"ACS data saved: {len(acs_df)} tracts → {acs_path}")
        paths["acs_data"] = acs_path

        # 3. Missingness audit
        self.logger.info("Running missingness audit on ACS data:")
        audit_df = _audit_missing(acs_df)
        audit_path = Path(self.config.paths.audit) / f"census_missing_audit_{census_config.acs_year}.csv"
        Path(self.config.paths.audit).mkdir(exist_ok=True)
        audit_df.to_csv(audit_path, index=False)
        paths["census_audit"] = audit_path

        # Summary
        self.logger.info(f"ACS Summary ({census_config.acs_year}):")
        self.logger.info(f"  Tracts: {len(acs_df)}")
        self.logger.info(
            f"  Median income range: "
            f"${acs_df['median_household_income'].min():,.0f} – "
            f"${acs_df['median_household_income'].max():,.0f}"
        )
        self.logger.info(
            f"  Poverty rate range: "
            f"{100*acs_df['poverty_rate'].min():.1f}% – "
            f"{100*acs_df['poverty_rate'].max():.1f}%"
        )

        return paths
