"""
CHIRTS-ERA5 Timeseries Extractor
==================================
Extracts daily Tmax or Tmin timeseries for a set of sites from CHIRTS-ERA5
annual NetCDF files. Supports two modes:

  - HTTP streaming  (default): streams files via byte-range HTTP — no download.
  - Local mode: reads from a directory of pre-downloaded annual NetCDF files,
                much faster for large date ranges or many sites.

The key efficiency: all sites sharing the same year are batched into a single
file open, so each annual NetCDF is opened at most once per product.

Input CSV format:
    site_name,longitude,latitude,start_date,end_date
    Site_A,-118.80764,37.17709,2017-01-01,2020-12-31
    Site_B,36.82,-1.28,2018-03-01,2021-06-30

Remote URL patterns:
    tmax: https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/tmax/netcdf/daily/CHIRTS-ERA5.daily_Tmax.{year}.nc
    tmin: https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/tmin/netcdf/daily/CHIRTS-ERA5.daily_Tmin.{year}.nc

Local filename patterns (files must follow the same naming convention):
    tmax: <local_dir>/CHIRTS-ERA5.daily_Tmax.{year}.nc
    tmin: <local_dir>/CHIRTS-ERA5.daily_Tmin.{year}.nc

Downloading files for local use (wget example):
    mkdir -p /data/chirts/tmax
    wget -P /data/chirts/tmax \\
        "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/tmax/netcdf/daily/CHIRTS-ERA5.daily_Tmax.{2017..2023}.nc"

Usage as a function:
    from chirts_era5_extract import extract_timeseries

    # HTTP streaming (no download)
    df = extract_timeseries(
        sites="my_sites.csv",
        product="tmax",
        combined=True,
        outdir="./output",
    )

    # Local files (faster for large runs)
    df = extract_timeseries(
        sites="my_sites.csv",
        product="tmax",
        local_dir="/data/chirts/tmax",
        combined=True,
        outdir="./output",
    )

Usage from the command line:
    python chirts_era5_extract.py --sites my_sites.csv --product tmax --combined
    python chirts_era5_extract.py --sites my_sites.csv --product tmax --local-dir /data/chirts/tmax --combined

Requirements:
    conda install -c conda-forge xarray h5netcdf fsspec aiohttp pandas tqdm
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import fsspec
import pandas as pd
import xarray as xr
from tqdm import tqdm
from datetime import datetime as dt

# ---------------------------------------------------------------------------
# Product configuration
# ---------------------------------------------------------------------------

PRODUCTS = {
    "tmax": {
        "url_template": (
            "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/"
            "tmax/netcdf/daily/CHIRTS-ERA5.daily_Tmax.{year}.nc"
        ),
        "filename_template": "CHIRTS-ERA5.daily_Tmax.{year}.nc",
        "description": "Max temperature (°C)",
    },
    "tmin": {
        "url_template": (
            "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/"
            "tmin/netcdf/daily/CHIRTS-ERA5.daily_Tmin.{year}.nc"
        ),
        "filename_template": "CHIRTS-ERA5.daily_Tmin.{year}.nc",
        "description": "Min temperature (°C)",
    },
}

ALL_PRODUCT_KEYS = list(PRODUCTS.keys())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Site CSV loading
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"site_name", "longitude", "latitude", "start_date", "end_date"}


def _load_sites(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Expected: site_name, longitude, latitude, start_date, end_date"
        )

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    bad = df[df[["longitude", "latitude"]].isna().any(axis=1)]
    if not bad.empty:
        log.warning(f"Dropping rows with invalid coordinates:\n{bad}")
        df = df.dropna(subset=["longitude", "latitude"])

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"]   = pd.to_datetime(df["end_date"])

    log.info(f"Loaded {len(df)} site(s) from {path}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build year → sites index
# ---------------------------------------------------------------------------

def _build_year_index(
    sites_df: pd.DataFrame,
) -> dict[int, list[dict]]:
    """
    Returns a dict mapping each year to the list of sites that need data
    from that year, along with the specific date slice required.

    Structure:
        {
            2017: [
                {"site_name": ..., "lon": ..., "lat": ...,
                 "start": Timestamp, "end": Timestamp},
                ...
            ],
            2018: [...],
        }

    Each site may appear in multiple years if its date range spans year
    boundaries. The date slice is clamped to the year so we only request
    the relevant time window when selecting from the dataset.
    """
    year_index: dict[int, list[dict]] = defaultdict(list)

    for _, row in sites_df.iterrows():
        start_year = row["start_date"].year
        end_year   = row["end_date"].year

        for year in range(start_year, end_year + 1):
            year_start = pd.Timestamp(year, 1, 1)
            year_end   = pd.Timestamp(year, 12, 31)

            # Clamp to the site's actual date range
            slice_start = max(row["start_date"], year_start)
            slice_end   = min(row["end_date"],   year_end)

            year_index[year].append({
                "site_name":  str(row["site_name"]).strip(),
                "lon":        float(row["longitude"]),
                "lat":        float(row["latitude"]),
                "start":      slice_start,
                "end":        slice_end,
            })

    return dict(sorted(year_index.items()))


# ---------------------------------------------------------------------------
# Open one annual NetCDF via byte-range HTTP
# ---------------------------------------------------------------------------

def _open_remote_nc(url: str) -> xr.Dataset | None:
    """
    Stream a remote NetCDF4 file via fsspec byte-range HTTP.
    Returns an open xarray Dataset, or None if the file is unreachable.
    """
    try:
        fs = fsspec.filesystem("https")
        f  = fs.open(url)
        ds = xr.open_dataset(f, engine="h5netcdf")
        return ds
    except Exception as exc:
        log.warning(f"  Could not open {url}: {exc}")
        return None


def _open_local_nc(path: Path) -> xr.Dataset | None:
    """
    Open a local NetCDF4 file. Returns an open xarray Dataset, or None
    if the file does not exist or cannot be opened.
    """
    if not path.exists():
        log.warning(f"  Local file not found: {path}")
        return None
    try:
        ds = xr.open_dataset(path, engine="h5netcdf")
        return ds
    except Exception as exc:
        log.warning(f"  Could not open {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Detect coordinate names flexibly
# ---------------------------------------------------------------------------

def _detect_coords(ds: xr.Dataset) -> tuple[str, str]:
    """
    Return (lat_name, lon_name) by matching common variants.
    Raises ValueError if either cannot be found.
    """
    lat_name = next(
        (c for c in ds.coords if c.lower() in ("lat", "latitude", "y")), None
    )
    lon_name = next(
        (c for c in ds.coords if c.lower() in ("lon", "longitude", "x")), None
    )
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Cannot detect lat/lon coordinates. "
            f"Available coords: {list(ds.coords)}"
        )
    return lat_name, lon_name


# ---------------------------------------------------------------------------
# Extract all sites from one open Dataset
# ---------------------------------------------------------------------------

def _extract_sites_from_year(
    ds: xr.Dataset,
    sites: list[dict],
    product_key: str,
) -> dict[str, pd.DataFrame]:
    """
    Given an open annual Dataset and a list of site dicts, extract a
    timeseries for each site and return {site_name: DataFrame}.
    Multiple sites sharing the same coordinates are deduplicated.
    """
    lat_name, lon_name = _detect_coords(ds)
    var = list(ds.data_vars)[0]
    col_name = product_key  # output column name matches product key

    results: dict[str, pd.DataFrame] = {}

    for site in sites:
        site_name = site["site_name"]
        try:
            # Select nearest grid cell then slice to the required date window
            ts = (
                ds.sel(
                    {lat_name: site["lat"], lon_name: site["lon"]},
                    method="nearest",
                )
                .sel(time=slice(site["start"], site["end"]))
            )

            df = (
                ts[var]
                .to_dataframe()[[var]]
                .rename(columns={var: col_name})
            )
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"

            results[site_name] = df

        except Exception as exc:
            log.error(f"  [{site_name}] Extraction failed: {exc}")

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_timeseries(
    sites: str | pd.DataFrame,
    product: str = "tmax",
    combined: bool = False,
    outdir: str | None = None,
    local_dir: str | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Extract daily CHIRTS-ERA5 timeseries for a set of sites.

    Each annual file is opened at most once, regardless of how many sites
    need data from that year.

    Parameters
    ----------
    sites : str or pd.DataFrame
        Path to a CSV file, or a DataFrame, with columns:
            site_name, longitude, latitude, start_date, end_date
        start_date / end_date format: 'YYYY-MM-DD'.

    product : str
        One of: 'tmax', 'tmin'. Default: 'tmax'.

    combined : bool
        If True  → return a single DataFrame with all sites stacked,
                   with site_name / longitude / latitude columns prepended.
        If False → return a dict of {site_name: DataFrame}.
        Default: False.

    outdir : str or None
        If provided, write output CSV(s) to this directory.
        combined=True  → all_sites_<product>_combined.csv
        combined=False → one <site_name>_<product>.csv per site.

    local_dir : str or None
        If provided, read annual NetCDF files from this local directory
        instead of streaming from the remote server. Files must follow
        the standard naming convention:
            CHIRTS-ERA5.daily_Tmax.{year}.nc  (tmax)
            CHIRTS-ERA5.daily_Tmin.{year}.nc  (tmin)
        This is significantly faster than HTTP streaming for large runs.
        If None (default), files are streamed via byte-range HTTP.

    Returns
    -------
    pd.DataFrame  (combined=True)
        Columns: site_name, longitude, latitude, <product>
        Index:   date (DatetimeIndex)

    dict[str, pd.DataFrame]  (combined=False)
        Keys: site names. Each DataFrame has a DatetimeIndex and one
        column named after the product.

    Raises
    ------
    ValueError
        If an unrecognised product is supplied or the CSV is malformed.

    Examples
    --------
    # HTTP streaming
    df = extract_timeseries("sites.csv", product="tmax", combined=True)

    # Local files (faster)
    df = extract_timeseries(
        "sites.csv",
        product="tmax",
        local_dir="/data/chirts/tmax",
        combined=True,
        outdir="./output",
    )
    """
    if product not in PRODUCTS:
        raise ValueError(
            f"Unknown product '{product}'. Choose from: {ALL_PRODUCT_KEYS}"
        )

    cfg = PRODUCTS[product]
    log.info(f"Product : {product} ({cfg['description']})")
    log.info(f"Source  : {'local (' + str(local_dir) + ')' if local_dir else 'HTTP streaming'}")

    # Load sites
    if isinstance(sites, pd.DataFrame):
        sites_df = sites.copy()
        sites_df.columns = sites_df.columns.str.strip().str.lower()
        sites_df["start_date"] = pd.to_datetime(sites_df["start_date"])
        sites_df["end_date"]   = pd.to_datetime(sites_df["end_date"])
    else:
        sites_df = _load_sites(sites)

    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    # Build year → sites index so each annual file is opened only once
    year_index = _build_year_index(sites_df)
    log.info(
        f"Date ranges span {len(year_index)} year(s): "
        f"{min(year_index)} – {max(year_index)}"
    )

    # Accumulate per-site results across all years
    # Structure: {site_name: [df_year1, df_year2, ...]}
    site_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for year, year_sites in tqdm(year_index.items(), desc="Years"):
        if local_dir:
            filename = cfg["filename_template"].format(year=year)
            file_path = Path(local_dir) / filename
            log.info(f"Opening {file_path}  ({len(year_sites)} site(s) need this year)")
            ds = _open_local_nc(file_path)
        else:
            url = cfg["url_template"].format(year=year)
            log.info(f"Opening {url}  ({len(year_sites)} site(s) need this year)")
            ds = _open_remote_nc(url)

        if ds is None:
            log.warning(f"  Skipping year {year} — file unavailable")
            continue

        try:
            year_results = _extract_sites_from_year(ds, year_sites, product)
            for site_name, df in year_results.items():
                if not df.empty:
                    site_frames[site_name].append(df)
        finally:
            ds.close()

    if not site_frames:
        log.error("No data extracted for any site.")
        return pd.DataFrame() if combined else {}

    # Concatenate annual chunks per site
    results: dict[str, pd.DataFrame] = {}
    for site_name, frames in site_frames.items():
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df.index.name = "date"

        if outdir and not combined:
            safe = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in site_name
            )
            out_path = Path(outdir) / f"{safe}_{product}.csv"
            df.to_csv(out_path)
            log.info(f"  Saved → {out_path}  ({len(df)} rows)")

        results[site_name] = df

    if not combined:
        return results

    # Assemble combined DataFrame with site metadata prepended
    frames = []
    for site_name, df in results.items():
        site_row = sites_df.loc[
            sites_df["site_name"].astype(str).str.strip() == site_name
        ].iloc[0]
        df = df.copy()
        df.insert(0, "site_name", site_name)
        df.insert(1, "longitude", float(site_row["longitude"]))
        df.insert(2, "latitude",  float(site_row["latitude"]))
        frames.append(df)

    combined_df = pd.concat(frames).sort_values(["site_name", "date"])

    if outdir:
        filename = f"all_sites_combined_{dt.now().strftime('%Y%m%dT%H%M')}.csv"
        out_path = Path(outdir) / filename
        combined_df.to_csv(out_path)
        log.info(
            f"Combined output saved → {out_path}  ({len(combined_df)} rows)"
        )

    return combined_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CHIRTS-ERA5 Tmax/Tmin timeseries from local files or via HTTP streaming.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
products available:
  tmax   Max temperature (°C)
  tmin   Min temperature (°C)

examples:
  # HTTP streaming, combined output
  python chirts_era5_extract.py --sites sites.csv --product tmax --combined --outdir ./output

  # Local files (faster), one CSV per site
  python chirts_era5_extract.py --sites sites.csv --product tmax --local-dir /data/chirts/tmax --outdir ./output

  # Local files, combined output
  python chirts_era5_extract.py --sites sites.csv --product tmin --local-dir /data/chirts/tmin --combined --outdir ./output
        """,
    )
    parser.add_argument(
        "--sites", required=True,
        help="CSV with columns: site_name, longitude, latitude, start_date, end_date",
    )
    parser.add_argument(
        "--product", default="tmax", choices=ALL_PRODUCT_KEYS,
        help="Product to extract (default: tmax)",
    )
    parser.add_argument(
        "--outdir", default="./chirts_output",
        help="Directory for output CSVs (default: ./chirts_output)",
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Write one combined CSV instead of one per site",
    )
    parser.add_argument(
        "--local-dir", default=None, dest="local_dir",
        help=(
            "Path to a local directory containing pre-downloaded annual NetCDF files. "
            "If omitted, files are streamed via byte-range HTTP."
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    result = extract_timeseries(
        sites=args.sites,
        product=args.product,
        combined=args.combined,
        outdir=args.outdir,
        local_dir=args.local_dir,
    )
    if isinstance(result, dict):
        log.info(f"Extraction complete. {len(result)} site(s) returned.")
    else:
        log.info(f"Extraction complete. {len(result)} total rows returned.")


if __name__ == "__main__":
    # --- edit these for your run ---
    result = extract_timeseries(
        sites="data/my_sites.csv",
        product="tmax",
        local_dir="/data/chirts/tmax",  # set to None to use HTTP streaming
        combined=True,
        outdir="data/",
    )