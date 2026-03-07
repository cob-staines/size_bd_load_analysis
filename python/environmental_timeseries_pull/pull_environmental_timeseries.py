
import sys
import os
sys.path.append(os.path.abspath("environmental_timeseries_pull/"))
import gee_chirps_chirts_pull as gee_point
import chirts_era5_extract

# CHIRPS for all dates (GEE)
results_chirps = gee_point.extract_timeseries(
    sites="environmental_timeseries_pull/inputs/site_coords_dates_all_2026-03-07.csv",
    products=["chirps"],
    combined=True,
    outdir="environmental_timeseries_pull/outputs/",
    project="amphibian-bd",
    chunk="decade",
    adaptive_retry=False,
)

# CHIRTS tmax through 2016 (GEE)
results_chirts_2016 = gee_point.extract_timeseries(
    sites="environmental_timeseries_pull/inputs/site_coords_dates_through_2016_2026-03-07.csv",
    products=["chirts_tmax"],
    combined=True,
    outdir="environmental_timeseries_pull/outputs/",
    project="amphibian-bd",
    chunk="decade",
    adaptive_retry=False,
)

# CHIRTS tmin through 2016 (GEE)
results_chirts_2016 = gee_point.extract_timeseries(
    sites="environmental_timeseries_pull/inputs/site_coords_dates_through_2016_2026-03-07.csv",
    products=["chirts_tmin"],
    combined=True,
    outdir="environmental_timeseries_pull/outputs/",
    project="amphibian-bd",
    chunk="decade",
    adaptive_retry=False,
)

# CHIRTS tmax and tmin 2017+ (ERA5)
# tmax
result_chirts_2017 = chirts_era5_extract.extract_timeseries(
    sites="environmental_timeseries_pull/inputs/site_coords_dates_2017_beyond_2026-03-07.csv",
    product="tmax",
    local_dir="environmental_timeseries_pull/chirts_era5/",
    combined=True,
    outdir="environmental_timeseries_pull/outputs/",
)
