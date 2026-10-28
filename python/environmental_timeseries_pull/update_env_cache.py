"""
Incremental CHIRPS / CHIRTS cache
=================================
Keeps a per-grid-cell cache of daily precipitation and max temperature, and
pulls only the cell-days that a site registry needs and the cache lacks.

Grid cells
    CHIRPS (GEE), CHIRTS (GEE) and CHIRTS-ERA5 (local NetCDF) share one grid:
    EPSG:4326, 0.05 deg cells with edges on multiples of 0.05 deg. A site's
    cell is (floor(lon / 0.05), floor(lat / 0.05)), and every value is pulled
    at the cell center. Pulling at the site point with getRegion(scale=5566)
    samples a grid that drifts from the native one (~0.0005 deg in California),
    so sites within ~50 m of a cell edge picked up the neighbouring cell.

Products (one parquet file per product in --cache-dir, columns cell_id, date,
value, source, pulled_at)
    chirps            precipitation (mm/day), GEE, all dates
    chirts_tmax       max temperature (C), GEE, through 2016
    chirts_era5_tmax  max temperature (C), local CHIRTS-ERA5 NetCDF, 2017+

A cell-date is "covered" once it has a cache row; rows with value NaN record
dates the product returned no data for, so they are not requested again.

Registry CSV (written by size_bd_unified/00_survey_data.Rmd and size_bd_gam/00_GAM_survey_data.Rmd)
    site, latitude, longitude, date_min, date_max
Each site needs [date_min - buffer_days, date_max]; a cell needs the union of its
sites' windows. For each registry, the site -> cell lookup is written next to it
as <registry>_cells.csv.

Usage (from this directory, conda env "ribbitr")
    # report what would be pulled for the unified and GAM pipelines
    python update_env_cache.py --registry ../../size_bd_unified/intermediates/site_registry.csv \
        ../../size_bd_gam/intermediates/site_registry_gam.csv --dry-run
    # pull it
    python update_env_cache.py --registry ../../size_bd_unified/intermediates/site_registry.csv \
        ../../size_bd_gam/intermediates/site_registry_gam.csv
    # one-off: seed the cache from an earlier site-based pull
    python update_env_cache.py --seed outputs/all_sites_combined_chirps_chirts_2026-09-11.csv
"""

import argparse
import datetime as dtm
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401  (parquet engine for the cache)
except ImportError:
    raise SystemExit("pyarrow not found: run from the conda env with the pull dependencies "
                     "(conda activate ribbitr)")

HERE = Path(__file__).resolve().parent

CELL_SIZE = 0.05

# cell-date ranges each product can supply (end None = open/available files)
PRODUCTS = {
    "chirps":           {"source": "gee",  "gee_key": "chirps",      "start": "1981-01-01", "end": None},
    "chirts_tmax":      {"source": "gee",  "gee_key": "chirts_tmax", "start": "1983-01-01", "end": "2016-12-31"},
    "chirts_era5_tmax": {"source": "era5", "era5_key": "tmax",       "start": "2017-01-01", "end": None},
}

# seed: sites closer than this to a cell edge are left out, as their GEE values
# may come from the neighbouring cell (see Grid cells above)
SEED_EDGE_TOL = 0.001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("update_env_cache")


# ---------------------------------------------------------------------------
# Grid cells
# ---------------------------------------------------------------------------

def cell_index(lon, lat):
    # round before floor so float noise (e.g. -118.55 / 0.05) cannot flip a cell
    cx = np.floor(np.round(np.asarray(lon, dtype=float) / CELL_SIZE, 9)).astype(int)
    cy = np.floor(np.round(np.asarray(lat, dtype=float) / CELL_SIZE, 9)).astype(int)
    return cx, cy


def cell_ids(lon, lat) -> np.ndarray:
    cx, cy = cell_index(lon, lat)
    return np.char.add(np.char.add(cx.astype(str), "_"), cy.astype(str))


def cell_center(cell_id: str) -> tuple[float, float]:
    cx, cy = (int(v) for v in cell_id.split("_"))
    return round((cx + 0.5) * CELL_SIZE, 6), round((cy + 0.5) * CELL_SIZE, 6)


def edge_distance(lon, lat) -> np.ndarray:
    fx = (np.asarray(lon, dtype=float) / CELL_SIZE) % 1
    fy = (np.asarray(lat, dtype=float) / CELL_SIZE) % 1
    return np.minimum.reduce([fx, 1 - fx, fy, 1 - fy]) * CELL_SIZE


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

CACHE_COLS = ["cell_id", "date", "value", "source", "pulled_at"]


def cache_path(cache_dir: Path, product: str) -> Path:
    return cache_dir / f"{product}.parquet"


def read_cache(cache_dir: Path, product: str) -> pd.DataFrame:
    path = cache_path(cache_dir, product)
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                             zip(CACHE_COLS, ["str", "datetime64[ns]", "float64", "str", "str"])})
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def append_cache(cache_dir: Path, product: str, new: pd.DataFrame) -> None:
    """Append rows, keeping the existing row for any cell-date already cached."""
    if new.empty:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    old = read_cache(cache_dir, product)
    out = (pd.concat([old, new[CACHE_COLS]], ignore_index=True)
             .drop_duplicates(["cell_id", "date"], keep="first")
             .sort_values(["cell_id", "date"]))
    tmp = cache_path(cache_dir, product).with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(cache_path(cache_dir, product))
    log.info(f"  {product}: +{len(out) - len(old)} rows -> {len(out)} cached")


# ---------------------------------------------------------------------------
# Requirements and gaps
# ---------------------------------------------------------------------------

def load_registry(path: Path, buffer_days: int) -> pd.DataFrame:
    reg = pd.read_csv(path, dtype={"site": str})
    missing = {"site", "latitude", "longitude", "date_min", "date_max"} - set(reg.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    dupes = reg["site"][reg["site"].duplicated()]
    if not dupes.empty:
        raise ValueError(f"{path} has duplicate sites: {sorted(dupes.unique())}")
    if reg[["latitude", "longitude"]].isna().any(axis=None):
        raise ValueError(f"{path} has sites without coordinates:\n"
                         f"{reg[reg[['latitude', 'longitude']].isna().any(axis=1)]}")
    reg["cell_id"] = cell_ids(reg["longitude"], reg["latitude"])
    reg["start"] = pd.to_datetime(reg["date_min"]) - pd.Timedelta(days=buffer_days)
    reg["end"] = pd.to_datetime(reg["date_max"])
    return reg


def write_site_cells(reg: pd.DataFrame, registry_path: Path) -> Path:
    out = registry_path.with_name(registry_path.stem + "_cells.csv")
    cells = reg[["site", "latitude", "longitude", "cell_id"]].copy()
    cells["cell_longitude"], cells["cell_latitude"] = zip(*cells["cell_id"].map(cell_center))
    cells["edge_distance_deg"] = edge_distance(reg["longitude"], reg["latitude"]).round(6)
    cells.to_csv(out, index=False)
    return out


def product_window(product: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    cfg = PRODUCTS[product]
    start = pd.Timestamp(cfg["start"])
    if cfg["end"]:
        end = pd.Timestamp(cfg["end"])
    elif cfg["source"] == "era5":
        years = sorted(int(p.stem.split(".")[-1]) for p in ERA5_DIR.glob("CHIRTS-ERA5.daily_Tmax.*.nc"))
        end = pd.Timestamp(years[-1], 12, 31) if years else start - pd.Timedelta(days=1)
    else:
        end = pd.Timestamp(dtm.date.today())
    return start, end


def find_gaps(need: pd.DataFrame, cached: pd.DataFrame, product: str) -> pd.DataFrame:
    """
    Contiguous runs of needed, uncached dates per cell: cell_id, start, end, n_days.
    need has one row per site window (cell_id, start, end).
    """
    p_start, p_end = product_window(product)
    have = cached.groupby("cell_id")["date"].agg(lambda d: set(d))
    runs = []
    for cell_id, windows in need.groupby("cell_id"):
        # union of the cell's site windows, clamped to the product's dates
        dates = pd.DatetimeIndex(sorted(set().union(*(
            pd.date_range(max(w.start, p_start), min(w.end, p_end), freq="D")
            for w in windows.itertuples(index=False)))))
        got = have.get(cell_id, set())
        missing = dates[~dates.isin(list(got))] if got else dates
        if missing.empty:
            continue
        # split into contiguous runs
        breaks = np.where(np.diff(missing.values).astype("timedelta64[D]").astype(int) > 1)[0]
        for seg in np.split(missing, breaks + 1):
            runs.append((cell_id, seg[0], seg[-1], len(seg)))
    return pd.DataFrame(runs, columns=["cell_id", "start", "end", "n_days"])


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

ERA5_DIR = HERE / "chirts_era5"


def _with_missing(cell_id: str, start, end, got: pd.Series, product: str) -> pd.DataFrame:
    """
    Cache rows for one request: returned values, plus NaN rows for dates the
    product returned nothing for. Dates after the last returned date are left
    out (not recorded), so data published later is picked up on a later run.
    """
    got = got.dropna()
    if got.empty:
        return pd.DataFrame(columns=CACHE_COLS)
    dates = pd.date_range(start, min(pd.Timestamp(end), got.index.max()), freq="D")
    out = pd.DataFrame({"cell_id": cell_id, "date": dates,
                        "value": got.reindex(dates).values.astype(float)})
    out["source"] = PRODUCTS[product]["source"]
    out["pulled_at"] = dtm.datetime.now().isoformat(timespec="seconds")
    return out


def fetch_gee(gaps: pd.DataFrame, product: str, cache_dir: Path, project: str,
              checkpoint: int = 25) -> None:
    import gee_chirps_chirts_pull as gee
    from tqdm import tqdm

    gee._init_gee(project=project)
    key = PRODUCTS[product]["gee_key"]
    buffer = []
    for i, row in enumerate(tqdm(gaps.itertuples(index=False), total=len(gaps), desc=product), 1):
        lon, lat = cell_center(row.cell_id)
        df = gee._fetch_product(row.cell_id, lon, lat,
                                row.start.strftime("%Y-%m-%d"), row.end.strftime("%Y-%m-%d"),
                                key, chunk="decade", adaptive_retry=True)
        if df.empty:
            log.warning(f"  {row.cell_id} {product}: nothing returned for "
                        f"{row.start.date()} - {row.end.date()}; left uncached")
            continue
        buffer.append(_with_missing(row.cell_id, row.start, row.end, df[key], product))
        if i % checkpoint == 0:
            append_cache(cache_dir, product, pd.concat(buffer, ignore_index=True))
            buffer = []
    if buffer:
        append_cache(cache_dir, product, pd.concat(buffer, ignore_index=True))


def fetch_era5(gaps: pd.DataFrame, product: str, cache_dir: Path) -> None:
    import chirts_era5_extract as era5

    key = PRODUCTS[product]["era5_key"]
    # one request name per gap, as the extractor keys results by name within a year
    gaps = gaps.reset_index(drop=True)
    names = gaps["cell_id"] + "|" + gaps.index.astype(str)
    sites = pd.DataFrame({
        "site_name": names,
        "longitude": [cell_center(c)[0] for c in gaps["cell_id"]],
        "latitude": [cell_center(c)[1] for c in gaps["cell_id"]],
        "start_date": gaps["start"],
        "end_date": gaps["end"],
    })
    results = era5.extract_timeseries(sites=sites, product=key, local_dir=str(ERA5_DIR), combined=False)
    frames = []
    for name, row in zip(names, gaps.itertuples(index=False)):
        df = results.get(name)
        if df is None or df.empty:
            log.warning(f"  {row.cell_id} {product}: nothing returned; left uncached")
            continue
        got = df[key].loc[row.start:row.end]
        frames.append(_with_missing(row.cell_id, row.start, row.end, got, product))
    if frames:
        append_cache(cache_dir, product, pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------
# Seeding from an earlier site-based pull
# ---------------------------------------------------------------------------

def seed(path: Path, cache_dir: Path) -> None:
    df = pd.read_csv(path, dtype={"site_name": str}, parse_dates=["date"], low_memory=False)
    sites = df[["site_name", "longitude", "latitude"]].drop_duplicates("site_name").copy()
    sites["cell_id"] = cell_ids(sites["longitude"], sites["latitude"])
    sites["edge_ok"] = edge_distance(sites["longitude"], sites["latitude"]) >= SEED_EDGE_TOL
    log.info(f"Seeding from {path.name}: {len(sites)} sites in {sites['cell_id'].nunique()} cells; "
             f"{(~sites['edge_ok']).sum()} sites within {SEED_EDGE_TOL} deg of a cell edge "
             f"are skipped for GEE products")
    df = df.merge(sites[["site_name", "cell_id", "edge_ok"]], on="site_name")
    stamp = f"seed:{path.name}"

    parts = {
        "chirps":           (df["edge_ok"], "precip_mm"),
        "chirts_tmax":      (df["edge_ok"] & (df["date"] < "2017-01-01"), "tmax_c"),
        "chirts_era5_tmax": (df["date"] >= "2017-01-01", "tmax_c"),
    }
    for product, (keep, col) in parts.items():
        sub = (df.loc[keep & df[col].notna(), ["cell_id", "date", col]]
                 .drop_duplicates(["cell_id", "date"])
                 .rename(columns={col: "value"}))
        sub["source"] = PRODUCTS[product]["source"]
        sub["pulled_at"] = stamp
        append_cache(cache_dir, product, sub)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", nargs="+", type=Path,
                        help="site registry CSV(s): site, latitude, longitude, date_min, date_max")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "env_cache")
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=list(PRODUCTS))
    parser.add_argument("--buffer-days", type=int, default=366,
                        help="days before each site's first sample to cover (default 366, for "
                             "the 30-day and 52-week windows)")
    parser.add_argument("--project", default="amphibian-bd", help="GEE cloud project")
    parser.add_argument("--dry-run", action="store_true", help="report gaps without pulling")
    parser.add_argument("--seed", type=Path, help="seed the cache from an earlier combined pull CSV")
    args = parser.parse_args()

    if args.seed:
        seed(args.seed, args.cache_dir)
    if not args.registry:
        return

    regs = []
    for path in args.registry:
        reg = load_registry(path, args.buffer_days)
        out = write_site_cells(reg, path)
        log.info(f"{path.name}: {len(reg)} sites in {reg['cell_id'].nunique()} cells "
                 f"(site -> cell lookup: {out.name})")
        regs.append(reg)
    reg = pd.concat(regs, ignore_index=True)
    need = reg[["cell_id", "start", "end"]]

    for product in args.products:
        cached = read_cache(args.cache_dir, product)
        gaps = find_gaps(need, cached, product)
        new_cells = set(gaps["cell_id"]) - set(cached["cell_id"])
        log.info(f"{product}: {len(gaps)} gap(s), {gaps['n_days'].sum() if len(gaps) else 0} cell-days, "
                 f"{gaps['cell_id'].nunique() if len(gaps) else 0} cells ({len(new_cells)} not yet cached)")
        if gaps.empty or args.dry_run:
            if args.dry_run and not gaps.empty:
                print(gaps.sort_values("n_days", ascending=False).head(20).to_string(index=False))
            continue
        if PRODUCTS[product]["source"] == "gee":
            fetch_gee(gaps, product, args.cache_dir, args.project)
        else:
            fetch_era5(gaps, product, args.cache_dir)


if __name__ == "__main__":
    main()
