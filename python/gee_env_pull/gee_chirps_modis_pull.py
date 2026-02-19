import yaml
import ee
import shapely.geometry as geom
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple, Union, Dict

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# load site_coords
sites = pd.read_csv(config['site_coords_path'])

# Authenticate once (run interactively once if needed), then initialize
try:
    ee.Initialize(project = config['gee_project'])
except Exception:
    ee.Authenticate()
    ee.Initialize(project = config['gee_project'])

# Accept polygon types: bbox tuple (minx, miny, maxx, maxy), GeoJSON mapping, or shapely geometry
PolyLike = Union[Tuple[float, float, float, float], dict, geom.base.BaseGeometry]


def to_ee_geometry(p: PolyLike) -> ee.Geometry:
    if isinstance(p, tuple) and len(p) == 4:
        minx, miny, maxx, maxy = p
        return ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    if isinstance(p, dict):
        return ee.Geometry(p)
    # shapely geometry
    return ee.Geometry(p.__geo_interface__)


def compute_chirps_average(
        sites: pd.DataFrame,
        scale: int = 5000,
        per_poly_export_threshold: int = 5000,
) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    """
    Always produce a daily timeseries.

    Expected columns in sites DataFrame:
      - W, S, E, N: bounding box coordinates
      - start_date: start date string for this site
      - end_date: end date string for this site

    Returns:
      - pivot: pandas.DataFrame indexed by date with columns = poly_id for polygons pulled client-side
      - exports: dict mapping poly_id -> export info for polygons exported to Drive
    """

    polygons = [
        (row['W'], row['S'], row['E'], row['N'])
        for _, row in sites.iterrows()
    ]

    rows_chirps = []
    exports_chirps = {}
    rows_modis = []
    exports_modis = {}

    for i, (idx, row) in tqdm(enumerate(sites.iterrows())):
        p = polygons[i]
        g = to_ee_geometry(p)

        start_date = row['start_date']
        end_date = row['end_date']

        # single-feature collection for this polygon
        fc = ee.FeatureCollection([ee.Feature(g, {"poly_id": i})])

        chirps_p = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .select("precipitation")
            .filterDate(start_date, end_date)
            .filterBounds(g)
        )

        modis_t = (
            ee.ImageCollection('MODIS/061/MOD11A1')
            .select('LST_Day_1km')
            .filterDate(start_date, end_date)
            .filterBounds(g)
        )

        # estimate number of images; if we cannot get a reliable count, force export
        try:
            num_images_chirps = int(chirps_p.size().getInfo())
        except Exception:
            num_images_chirps = None

        try:
            num_images_modis = int(modis_t.size().getInfo())
        except Exception:
            num_images_modis = None


        # map reducer over images (server-side)
        def per_image(img):
            date = img.date().format("YYYY-MM-dd")
            reduced = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale)
            def set_date(f):
                return f.set("date", date)
            return reduced.map(set_date)

        # if estimated feature count for this polygon is small, pull to client
        if num_images_chirps is not None and num_images_chirps <= per_poly_export_threshold:
            chirps_reduced_fc = chirps_p.map(per_image).flatten()

            try:
                features = chirps_reduced_fc.getInfo().get("features", [])
            except Exception:
                features = []
            for f in features:
                props = f.get("properties", {})
                pid = props.get("poly_id")
                date = props.get("date")
                value = None
                for k, v in props.items():
                    if k in ("poly_id", "date"):
                        continue
                    if isinstance(v, (int, float)):
                        value = v
                        break
                rows_chirps.append({"poly_id": pid, "date": date, "value": value})
        else:
            # start an export for large per-polygon timeseries
            task = ee.batch.Export.table.toDrive(
                collection=chirps_reduced_fc,
                description=f"chirps_timeseries_poly_{i}",
                folder=config.get("export_folder", "gee_exports"),
                fileFormat="CSV",
            )
            task.start()
            exports_chirps[i] = {"task_id": task.id, "estimated_features": num_images_chirps}

        if num_images_modis is not None and num_images_modis <= per_poly_export_threshold:
            modis_reduced_fc = modis_t.map(per_image).flatten()

            try:
                features = modis_reduced_fc.getInfo().get("features", [])
            except Exception:
                features = []
            for f in features:
                props = f.get("properties", {})
                pid = props.get("poly_id")
                date = props.get("date")
                value = None
                for k, v in props.items():
                    if k in ("poly_id", "date"):
                        continue
                    if isinstance(v, (int, float)):
                        value = v
                        break
                rows_modis.append({"poly_id": pid, "date": date, "value": value})
        else:
            # start an export for large per-polygon timeseries
            task = ee.batch.Export.table.toDrive(
                collection=modis_reduced_fc,
                description=f"modis_timeseries_poly_{i}",
                folder=config.get("export_folder", "gee_exports"),
                fileFormat="CSV",
            )
            task.start()
            exports_modis[i] = {"task_id": task.id, "estimated_features": num_images_modis}

    # assemble dataframe from pulled polygons
    df_chirps = pd.DataFrame(rows_chirps)
    df_modis = pd.DataFrame(rows_modis)

    if df_chirps.empty:
        pivot_chirps = pd.DataFrame(columns=["poly_id", "date", "value"])
    else:
        df_chirps["value"] = pd.to_numeric(df_chirps["value"], errors="coerce")
        df_chirps["date"] = pd.to_datetime(df_chirps["date"])
        pivot_chirps = df_chirps.pivot_table(index="date", columns="poly_id", values="value", aggfunc="mean")

    if df_modis.empty:
        pivot_modis = pd.DataFrame(columns=["poly_id", "date", "value"])
    else:
        df_modis["value"] = pd.to_numeric(df_modis["value"], errors="coerce")
        df_modis["date"] = pd.to_datetime(df_modis["date"])
        pivot_modis = df_modis.pivot_table(index="date", columns="poly_id", values="value", aggfunc="mean")


    return pivot_chirps, pivot_modis, exports_chirps, exports_modis


# Example usage
if __name__ == "__main__":
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # load site_coords
    sites = pd.read_csv(config['site_coords_path'])
    sites['site_index'] = range(len(sites))

    # # Get a single average per polygon across the date range
    # avg_per_poly = compute_chirps_average(polygons, start, end, timeseries=False)
    # print("Average daily rainfall (mm/day) per polygon id:", avg_per_poly)

    # Or get a daily timeseries per polygon (pandas DataFrame)
    ts = compute_chirps_average(sites)
    ts_chirps = pd.DataFrame(ts[0]).reset_index()
    ts_modis = pd.DataFrame(ts[1]).reset_index()

    ts_long_chirps = pd.melt(ts_chirps, id_vars='date', var_name='site_index', value_name='precipitation_mm').dropna(subset = ['precipitation_mm'])
    ts_long_modis = pd.melt(ts_modis, id_vars='date', var_name='site_index', value_name='temperature_c').dropna(subset = ['temperature_c'])
    ts_long_modis['temperature_c'] = ts_long_modis['temperature_c'] * 0.02 - 273.15  # scale and convert to Celsius
    ts_out = ts_long_chirps.merge(ts_long_modis, on=['date', 'site_index'], how="outer").merge(sites[['site_index', 'site']], on='site_index')[['site', 'date', 'precipitation_mm', 'temperature_c']]
    ts_out.to_csv(config['output_path'], index=False)
