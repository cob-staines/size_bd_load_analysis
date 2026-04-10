
import sys
import os
sys.path.append(os.path.abspath("environmental_timeseries_pull/"))
import gee_chirps_chirts_pull as gee_point
import chirts_era5_extract
import pandas as pd

# CHIRPS for all dates (GEE)
results_chirps = gee_point.extract_timeseries(
    sites="environmental_timeseries_pull/inputs/site_date_all_2026-03-24.csv",
    products=["chirps"],
    combined=True,
    outdir="environmental_timeseries_pull/outputs/",
    project="amphibian-bd",
    chunk="decade",
    adaptive_retry=False,
)

# CHIRTS tmax through 2016 (GEE)
results_chirts_pre_2017 = gee_point.extract_timeseries(
    sites="environmental_timeseries_pull/inputs/site_date_pre_2017_2026-03-24.csv",
    products=["chirts_tmax"],
    combined=True,
    outdir="environmental_timeseries_pull/outputs/",
    project="amphibian-bd",
    chunk="decade",
    adaptive_retry=False,
)

# CHIRTS tmax and tmin 2017+ (ERA5)
# tmax
result_chirts_post_2017 = chirts_era5_extract.extract_timeseries(
    sites="environmental_timeseries_pull/inputs/site_date_post_2017_2026-03-24.csv",
    product="tmax",
    local_dir="environmental_timeseries_pull/chirts_era5/",
    combined=True,
    outdir="environmental_timeseries_pull/outputs/",
)

# combine outputs
results_chirps = pd.read_csv("environmental_timeseries_pull/outputs/all_sites_combined_20260324T1801.csv",
                             parse_dates=["date"],
                             dtype={"site_name": str}).rename(columns={"chirps": "precip_mm"})
results_chirts_pre_2017 = pd.read_csv("environmental_timeseries_pull/outputs/all_sites_combined_20260325T0933.csv",
                             parse_dates=["date"],
                             dtype={"site_name": str}).rename(columns={"chirts_tmax": "tmax_c"})
results_chirts_post_2017 = pd.read_csv("environmental_timeseries_pull/outputs/all_sites_combined_20260325T0943.csv",
                             parse_dates=["date"],
                             dtype={"site_name": str}).rename(columns={"tmax": "tmax_c"})

results_chirts = pd.concat([results_chirts_pre_2017,
                            results_chirts_post_2017],
                           ignore_index=True)

results_all = results_chirps.merge(results_chirts, on=["site_name", "date", "latitude", "longitude"], how="outer")

check_1 = results_all[results_all["precip_mm"].isna()]
check_2 = results_all[results_all["tmax_c"].isna()]

results_all.to_csv("environmental_timeseries_pull/outputs/all_sites_combined_chirps_chirts_2026-03-25.csv", index = False)