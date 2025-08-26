#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterstats import point_query
import numpy as np
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shp_shape


JOB_ID = "compare_to_hwms"


def convert_feet(feet: float, unit_guess: str) -> float:
    """Convert feet to CRS units."""
    if unit_guess == 'meters':
        return feet * 0.3048
    elif unit_guess == 'feet':
        return feet
    else:
        return feet * 0.3048  # Assume meters if unknown


def process_points(vector_path: str, raster_path: str, output_path: str, log: logging.Logger):
    """Process point data against raster to compute hits and miss distances."""

    # Load raster metadata
    log.info("Loading raster to read CRS and resolution...")
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        res_x, res_y = src.res
        log.info(f"Raster CRS: {raster_crs}")
        log.info(f"Raster resolution: {res_x} x {res_y} units/pixel")

        # Determine CRS units
        if raster_crs.is_projected:
            if raster_crs.linear_units == 'metre':
                unit_guess = 'meters'
            elif raster_crs.linear_units == 'foot':
                unit_guess = 'feet'
            else:
                unit_guess = 'meters'  # Assume meters for unknown projected units
                log.warning(f"Unknown projected CRS units: {raster_crs.linear_units}. Assuming meters.")
        else:
            unit_guess = 'degrees'
            log.warning("Raster CRS is geographic (e.g., degrees). Distances may be inaccurate without reprojection.")

        log.info(f"CRS units determined as: {unit_guess}")

    # Load vector data
    log.info(f"Loading vector data from {vector_path}...")
    gdf = gpd.read_file(vector_path)
    log.info(f"Loaded {len(gdf)} points")
    log.info(f"Original vector CRS: {gdf.crs}")

    # Reproject to raster CRS
    gdf_proj = gdf.to_crs(raster_crs)

    # Sample raster at point locations
    log.info("Sampling raster values at point locations...")
    point_vals = point_query(gdf_proj, raster_path, interpolate='nearest')

    gdf_proj['raster_val'] = point_vals
    gdf_proj['hit'] = gdf_proj['raster_val'].apply(
        lambda x: "hit" if (x is not None and x > 0) else "miss"
    )

    log.info(f"Points with 'hit': {(gdf_proj['hit'] == 'hit').sum()} / {len(gdf_proj)}")

    # --- Compute distance to nearest positive feature edge using vectorization ---
    log.info("Computing distance-to-hit using vectorized positive areas...")

    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
        positive_mask = (data > 0)
        transform = src.transform

        # Extract shapes of positive areas
        shape_gen = rio_shapes(data.astype(np.int16), mask=positive_mask, transform=transform)
        geoms = [shp_shape(geom) for geom, val in shape_gen if val > 0]

        log.info(f"Number of positive feature geometries: {len(geoms)}")

        # Function to compute min distance to any positive geometry
        def calc_dist(geom):
            if not geoms:
                return None
            return min(geom.distance(g) for g in geoms)

    gdf_proj['dist_to_hit_crs'] = gdf_proj.geometry.apply(calc_dist)

    # Convert CRS distance to feet
    if unit_guess == 'meters':
        gdf_proj['dist_to_hit_ft'] = gdf_proj['dist_to_hit_crs'] / 0.3048
    elif unit_guess == 'feet':
        gdf_proj['dist_to_hit_ft'] = gdf_proj['dist_to_hit_crs']
    else:
        log.warning("Distances are in degrees; conversion to feet is not applicable. Keeping CRS units.")
        gdf_proj['dist_to_hit_ft'] = gdf_proj['dist_to_hit_crs']

    # Only keep distances for misses (set hits to 0)
    gdf_proj.loc[gdf_proj['hit'] == 'hit', 'dist_to_hit_ft'] = 0.0

    # Clean up temp columns
    gdf_proj = gdf_proj.drop(columns=['raster_val', 'dist_to_hit_crs'])

    # Save
    log.info(f"Saving updated GeoPackage to {output_path}...")
    gdf_proj.to_file(output_path, driver='GPKG')
    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute raster hits and distance to hits for point data."
    )
    parser.add_argument("--vector_path", required=True, help="Input vector file (e.g., GeoPackage with points).")
    parser.add_argument("--raster_path", required=True, help="Input raster file (GeoTIFF).")
    parser.add_argument("--output_path", required=True, help="Output GeoPackage with results.")
    args = parser.parse_args()

    log = logging.getLogger(JOB_ID)

    try:
        process_points(args.vector_path, args.raster_path, args.output_path, log)
        log.info("Processing completed successfully.")
    except Exception as e:
        log.error(f"{JOB_ID} run failed: {type(e).__name__}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()