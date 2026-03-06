"""
GEE CHIRPS / CHIRTS Timeseries Extractor
=========================================
Pulls daily precipitation (CHIRPS) and/or temperature (CHIRTS Tmax/Tmin)
for a set of sites from Google Earth Engine.

Can be used as an importable function or run directly from the command line.

Input CSV format (--sites):
    site_name,longitude,latitude,start_date,end_date
    Site_A,36.82,-1.28,2000-01-01,2020-12-31
    Site_B,-87.65,41.85,2005-06-01,2015-12-31

Usage as a function:
    from gee_chirps_chirts_extract import extract_timeseries

    df = extract_timeseries(
        sites="my_sites.csv",
        products=["chirps", "chirts_tmax"],   # any combination
        combined=True,
        chunk="none",        # attempt full date range in one GEE call
        adaptive_retry=True, # subdivide automatically if it fails
    )

Usage from the command line:
    python gee_chirps_chirts_extract.py --sites my_sites.csv --outdir ./output
    python gee_chirps_chirts_extract.py --sites my_sites.csv --products chirps chirts_tmin --combined
    python gee_chirps_chirts_extract.py --sites my_sites.csv --chunk none --adaptive-retry --combined

Requirements:
    pip install earthengine-api pandas tqdm python-dateutil
    earthengine authenticate   # once, to set up credentials
"""

import argparse
import logging
from pathlib import Path

import ee
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# GEE dataset configuration
# ---------------------------------------------------------------------------
PRODUCTS = {
    "chirps": {
        "collection": "UCSB-CHG/CHIRPS/DAILY",
        "band": "precipitation",
        "scale": 5566,
        "description": "Precipitation (mm/day)",
    },
    "chirts_tmax": {
        "collection": "UCSB-CHG/CHIRTS/DAILY",
        "band": "maximum_temperature",
        "scale": 5566,
        "description": "Max temperature (°C)",
    },
    "chirts_tmin": {
        "collection": "UCSB-CHG/CHIRTS/DAILY",
        "band": "minimum_temperature",
        "scale": 5566,
        "description": "Min temperature (°C)",
    },
    "chirts_rh": {
        "collection": "UCSB-CHG/CHIRTS/DAILY",
        "band": "relative_humidity",
        "scale": 5566,
        "description": "Relative humidity (%)",
    },
}

ALL_PRODUCT_KEYS = list(PRODUCTS.keys())

# Ordered coarsest → finest; adaptive retry descends this ladder one step at a time.
# "none" means no chunking — the full date range is sent as a single GEE request.
CHUNK_LADDER = ["none", "decade", "year", "month", "week", "day"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GEE initialisation (called once)
# ---------------------------------------------------------------------------

_gee_initialised = False


def _init_gee(project: str | None = None) -> None:
    """Initialise GEE, authenticating interactively if needed."""
    global _gee_initialised
    if _gee_initialised:
        return
    log.info("Initialising Google Earth Engine...")
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:
        log.info("GEE credentials not found — launching authentication flow...")
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()
    log.info("GEE initialised.")
    _gee_initialised = True


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

    log.info(f"Loaded {len(df)} site(s) from {path}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Date chunking helper
# ---------------------------------------------------------------------------

def _date_chunks(
    start_date: str,
    end_date: str,
    chunk: str = "year",
) -> list[tuple[str, str]]:
    """
    Split [start_date, end_date) into (chunk_start, chunk_end) pairs.
    chunk_end is exclusive, matching GEE filterDate convention.

    chunk="none" returns a single pair covering the full date range.
    """
    from dateutil.relativedelta import relativedelta

    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)

    # "none" = no chunking; pass the full range as a single request
    if chunk == "none":
        return [(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))]

    delta = {
        "decade":  relativedelta(years=10),
        "year":    relativedelta(years=1),
        "month":   relativedelta(months=1),
        "week":    relativedelta(weeks=1),
        "day":     relativedelta(days=1),
    }.get(chunk)

    if delta is None:
        raise ValueError(f"chunk must be one of {CHUNK_LADDER}, got '{chunk}'")

    periods, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + delta, end)
        periods.append((
            cursor.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        cursor = chunk_end

    return periods


# ---------------------------------------------------------------------------
# Single-chunk GEE fetch
# ---------------------------------------------------------------------------

def _getregion(
    point: ee.Geometry.Point,
    cfg: dict,
    chunk_start: str,
    chunk_end: str,
) -> pd.DataFrame | None:
    """
    Execute one getRegion call for [chunk_start, chunk_end).
    Returns a tidy DataFrame on success, None on any failure or empty result.
    The value column is named 'value' (renamed to product_key by the caller).
    """
    col = (
        ee.ImageCollection(cfg["collection"])
        .filterDate(chunk_start, chunk_end)
        .select(cfg["band"])
    )

    try:
        region_data = col.getRegion(point, cfg["scale"]).getInfo()
    except Exception as exc:
        log.debug(f"    getRegion raised: {exc}")
        return None

    if not region_data or len(region_data) < 2:
        return None

    header = region_data[0]
    rows   = region_data[1:]
    df = pd.DataFrame(rows, columns=header)
    df["date"] = pd.to_datetime(df["time"], unit="ms").dt.normalize()
    df = (
        df[["date", cfg["band"]]]
        .rename(columns={cfg["band"]: "value"})
        .set_index("date")
        .sort_index()
    )
    # Return the DataFrame even if values are null — the caller checks for
    # all-null responses and decides whether to retry. Returning None here
    # is reserved for request-level failures (exception or empty response).
    return df if not df.empty else None


# ---------------------------------------------------------------------------
# Per-site, per-product extraction (with optional adaptive retry)
# ---------------------------------------------------------------------------

def _fetch_product(
    site_name: str,
    lon: float,
    lat: float,
    start_date: str,
    end_date: str,
    product_key: str,
    chunk: str = "year",
    adaptive_retry: bool = False,
) -> pd.DataFrame:
    """
    Fetch a single product timeseries for one site.

    Parameters
    ----------
    chunk : str
        Starting chunk granularity. One of: none, decade, year, month, week, day.
        "none" sends the full date range as a single GEE request (fastest, but
        most likely to fail on data gaps).
    adaptive_retry : bool
        If True, a chunk is retried with the next finer granularity when either:
          - the GEE request fails outright, or
          - the request succeeds but returns all-null values (silent data gap,
            which can occur for some products even when others return fine data
            for the same period, e.g. CHIRTS gaps that don't affect CHIRPS).
        Day-level failures are logged and skipped — they represent genuinely
        missing data. If False, failed/null chunks are skipped with a warning.
    """
    cfg   = PRODUCTS[product_key]
    point = ee.Geometry.Point([lon, lat])
    frames: list[pd.DataFrame] = []

    def _fetch_chunk_with_retry(c_start: str, c_end: str, current_chunk: str) -> None:
        """
        Attempt to fetch [c_start, c_end). Retries with finer granularity if:
          - the request fails outright (None returned), or
          - the request succeeds but all values are null (silent data gap).
        Successful DataFrames (with at least one non-null value) are appended
        to the outer `frames` list with null rows dropped.
        """
        result = _getregion(point, cfg, c_start, c_end)

        # Determine whether to retry: None = request failed; all-null = silent gap
        if result is not None and result["value"].notna().any():
            # At least some valid data — keep it, dropping any null rows
            frames.append(
                result.dropna(subset=["value"])
                      .rename(columns={"value": product_key})
            )
            return

        # ---- chunk failed or returned all nulls ----
        reason = "request failed" if result is None else "all values null"

        if not adaptive_retry:
            log.warning(
                f"  [{site_name}] {product_key}: {current_chunk}-chunk "
                f"{c_start}→{c_end} {reason} — skipping "
                f"(use adaptive_retry=True to subdivide automatically)"
            )
            return

        # Find the next finer granularity on the ladder
        try:
            next_chunk = CHUNK_LADDER[CHUNK_LADDER.index(current_chunk) + 1]
        except IndexError:
            # Already at 'day' — no finer level available
            log.warning(
                f"  [{site_name}] {product_key}: day-level chunk "
                f"{c_start}→{c_end} {reason} — genuinely missing data, skipping"
            )
            return

        log.info(
            f"  [{site_name}] {product_key}: {current_chunk}-chunk "
            f"{c_start}→{c_end} {reason} → retrying as {next_chunk} sub-chunks"
        )
        for sub_start, sub_end in _date_chunks(c_start, c_end, next_chunk):
            _fetch_chunk_with_retry(sub_start, sub_end, next_chunk)

    for chunk_start, chunk_end in _date_chunks(start_date, end_date, chunk):
        _fetch_chunk_with_retry(chunk_start, chunk_end, chunk)

    if not frames:
        log.warning(
            f"  [{site_name}] {product_key}: no data returned across all chunks"
        )
        return pd.DataFrame()

    result = pd.concat(frames)
    result = result[~result.index.duplicated(keep="first")]
    return result.sort_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_timeseries(
    sites: str | pd.DataFrame,
    products: list[str] | None = None,
    combined: bool = False,
    outdir: str | None = None,
    project: str | None = None,
    chunk: str = "none",
    adaptive_retry: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Extract daily CHIRPS / CHIRTS timeseries from Google Earth Engine.

    Parameters
    ----------
    sites : str or pd.DataFrame
        Path to a CSV file, or a DataFrame, with columns:
            site_name, longitude, latitude, start_date, end_date
        Each row is one site. start_date / end_date format: 'YYYY-MM-DD'.

    products : list of str, optional
        Any combination of: 'chirps', 'chirts_tmax', 'chirts_tmin', 'chirts_rh'.
        Defaults to all four if not specified.

    combined : bool, optional
        If True  → return (and optionally save) a single DataFrame with all sites.
        If False → return a dict of {site_name: DataFrame}, one per site.
        Default: False.

    outdir : str or None, optional
        If provided, write output CSV(s) to this directory.
        In combined mode → one file: all_sites_combined.csv
        In per-site mode → one file per site: <site_name>.csv

    project : str or None, optional
        GEE cloud project ID. Required for some authentication setups.

    chunk : str, optional
        Starting granularity for GEE requests. One of:
            'none' (default), 'decade', 'year', 'month', 'week', 'day'
        'none' sends the full date range as a single GEE request — fastest
        when it works. Use 'year' or 'month' for more robust extraction
        without adaptive retry.

    adaptive_retry : bool, optional
        If True, any chunk that fails is automatically re-attempted using the
        next finer granularity on the ladder:
            none → decade → year → month → week → day
        Each failed chunk is subdivided and retried independently, so a single
        bad month won't discard an entire year. Day-level failures are treated
        as genuinely missing data and skipped.
        Default: False.

    Returns
    -------
    pd.DataFrame  (if combined=True)
        Columns: site_name, longitude, latitude, then one column per requested
                 product. Index: date (DatetimeIndex).

    dict[str, pd.DataFrame]  (if combined=False)
        Keys are site names; each DataFrame has a DatetimeIndex and one column
        per requested product.

    Raises
    ------
    ValueError
        If an unrecognised product key or chunk value is supplied, or the CSV
        is malformed.

    Examples
    --------
    # Start with a single unchunked request, retry failures automatically
    results = extract_timeseries(
        "sites.csv",
        chunk="none",
        adaptive_retry=True,
    )

    # Precipitation only, yearly chunks, no retry, save combined CSV
    df = extract_timeseries(
        "sites.csv",
        products=["chirps"],
        combined=True,
        outdir="./output",
        chunk="year",
    )
    """
    # --- validate inputs ---
    if products is None:
        products = ALL_PRODUCT_KEYS
    else:
        bad = [p for p in products if p not in PRODUCTS]
        if bad:
            raise ValueError(
                f"Unknown product(s): {bad}. "
                f"Valid options are: {ALL_PRODUCT_KEYS}"
            )

    if chunk not in CHUNK_LADDER:
        raise ValueError(f"chunk must be one of {CHUNK_LADDER}, got '{chunk}'")

    product_desc = ", ".join(
        f"{k} ({PRODUCTS[k]['description']})" for k in products
    )
    log.info(f"Products requested : {product_desc}")
    log.info(f"Chunk size         : {chunk}")
    log.info(f"Adaptive retry     : {adaptive_retry}")

    # --- initialise GEE ---
    _init_gee(project=project)

    # --- load sites ---
    if isinstance(sites, pd.DataFrame):
        sites_df = sites.copy()
        sites_df.columns = sites_df.columns.str.strip().str.lower()
    else:
        sites_df = _load_sites(sites)

    # --- output directory ---
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    # --- extraction loop ---
    results: dict[str, pd.DataFrame] = {}

    for _, row in tqdm(sites_df.iterrows(), total=len(sites_df), desc="Sites"):
        site_name  = str(row["site_name"]).strip()
        lon        = float(row["longitude"])
        lat        = float(row["latitude"])
        start_date = str(row["start_date"]).strip()
        end_date   = str(row["end_date"]).strip()

        log.info(
            f"Processing site: {site_name}  ({lat}, {lon})  "
            f"{start_date} → {end_date}"
        )

        frames = []
        for key in products:
            try:
                df = _fetch_product(
                    site_name, lon, lat, start_date, end_date, key,
                    chunk=chunk,
                    adaptive_retry=adaptive_retry,
                )
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                log.error(f"  [{site_name}] Error fetching {key}: {exc}")

        if not frames:
            log.warning(f"  [{site_name}] No data returned — skipping.")
            continue

        site_df = frames[0]
        for df in frames[1:]:
            site_df = site_df.join(df, how="outer")

        site_df.index.name = "date"

        if outdir and not combined:
            safe = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in site_name
            )
            out_path = Path(outdir) / f"{safe}.csv"
            site_df.to_csv(out_path)
            log.info(f"  Saved → {out_path}  ({len(site_df)} rows)")

        results[site_name] = site_df

    if not results:
        log.error("No data extracted for any site.")
        return pd.DataFrame() if combined else {}

    # --- assemble combined output ---
    if combined:
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
            out_path = Path(outdir) / "all_sites_combined.csv"
            combined_df.to_csv(out_path)
            log.info(
                f"Combined output saved → {out_path}  ({len(combined_df)} rows)"
            )

        return combined_df

    return results


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CHIRPS/CHIRTS timeseries from GEE for a list of sites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
products available:
  chirps        Precipitation (mm/day)       UCSB-CHG/CHIRPS/DAILY
  chirts_tmax   Max temperature (°C)         UCSB-CHG/CHIRTS/DAILY
  chirts_tmin   Min temperature (°C)         UCSB-CHG/CHIRTS/DAILY
  chirts_rh     Relative humidity (%)        UCSB-CHG/CHIRTS/DAILY

chunk ladder (coarse → fine):
  {' → '.join(CHUNK_LADDER)}
  ("none" = no chunking, full date range in one GEE call)

examples:
  # No chunking, adaptive retry on failure (recommended default)
  python gee_chirps_chirts_extract.py --sites sites.csv --adaptive-retry --combined

  # All products, yearly chunks, one CSV per site
  python gee_chirps_chirts_extract.py --sites sites.csv --chunk year --outdir ./output

  # Precipitation only, combined output
  python gee_chirps_chirts_extract.py --sites sites.csv --products chirps --combined

  # Temperature only, monthly chunks with adaptive retry
  python gee_chirps_chirts_extract.py --sites sites.csv --products chirts_tmax chirts_tmin --chunk month --adaptive-retry
        """,
    )
    parser.add_argument(
        "--sites", required=True,
        help="CSV with columns: site_name, longitude, latitude, start_date, end_date",
    )
    parser.add_argument(
        "--outdir", default="./gee_output",
        help="Directory for output CSVs (default: ./gee_output)",
    )
    parser.add_argument(
        "--products", nargs="+", default=ALL_PRODUCT_KEYS,
        choices=ALL_PRODUCT_KEYS, metavar="PRODUCT",
        help=f"Products to extract (default: all). Choices: {ALL_PRODUCT_KEYS}",
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Write one combined CSV instead of one per site",
    )
    parser.add_argument(
        "--chunk", default="none", choices=CHUNK_LADDER,
        help="Starting chunk granularity for GEE requests (default: none)",
    )
    parser.add_argument(
        "--adaptive-retry", action="store_true",
        help=(
            "If a chunk fails, automatically retry with the next finer "
            "granularity down to day-level"
        ),
    )
    parser.add_argument(
        "--project", default=None,
        help="GEE cloud project ID (required for some auth setups)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    result = extract_timeseries(
        sites=args.sites,
        products=args.products,
        combined=args.combined,
        outdir=args.outdir,
        project=args.project,
        chunk=args.chunk,
        adaptive_retry=args.adaptive_retry,
    )
    if isinstance(result, dict):
        log.info(f"Extraction complete. {len(result)} site(s) returned.")
    else:
        log.info(f"Extraction complete. {len(result)} rows returned.")


if __name__ == "__main__":
    results = extract_timeseries(
        sites="data/site_coords_dates_2026-03-05_test.csv",
        products=["chirps", "chirts_tmax", "chirts_tmin", "chirts_rh"],
        combined=True,
        outdir="data/",
        project="amphibian-bd",
        chunk="decade",
        adaptive_retry=False,
    )