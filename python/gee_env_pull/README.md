# CHIRPS & MODIS daily region pull

A simple python script to pull daily precipitation (CHIRPS) and daytime temperature (MODIS) for a date range, calculated for a list of sites with corresponding N/S/E/W coordinates (averaged spatially).

## Getting Started

This script uses Google Earth Engine (GEE), and requires logging in with a GEE-linked google account

1. prepare the site extent coordinates (lat/lon for N/S/E/W extent) for each site, as in the example site_coordinates file (`site_coords_example.csv`).
2. Fill out the configuration file (`config.yml`) with dates and project details
3. run `gee_chirps_modis_pull.py`

## Output

[*CHIRPS*](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY) -- Daily precipitation (mm) estimates averaged over the site extent, for each site

[*MODIS*](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1) -- Daile daytime surface temperature (C) estimates averaged over the site extent, for each site

## Note

Queries are submitted seperately to GEE for each site, for CHIRPS and MODIS. Queries which include more than 5000 results (in space or time) will fail (try shorter timespans or smaller areas).

Pull requests welcome :)