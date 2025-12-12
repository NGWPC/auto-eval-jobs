import numpy as np
from osgeo import gdal, osr

srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)

# Define geotransform parameters
# Format: (top_left_x, pixel_width, rotation, top_left_y, rotation, pixel_height)
# Starting at longitude 0, latitude 45, with 0.001-degree pixel size
x_min = 0.0
y_max = 45.0
pixel_size = 0.001  # approximately 111 meters at the equator
geotransform = (x_min, pixel_size, 0, y_max, 0, -pixel_size)

# Corner definitions used by both extent and depth rasters
# Each raster has valid data in one corner, allowing mosaic to combine them
corner_tiles = [
    {"row": (0, 256), "col": (0, 256)},  # Top-left corner
    {"row": (0, 256), "col": (512, 768)},  # Top-right corner
    {"row": (512, 768), "col": (0, 256)},  # Bottom-left corner
    {"row": (512, 768), "col": (512, 768)},  # Bottom-right corner
]


def create_extent_rasters():
    """Create binary extent rasters (uint8) with values 0/1."""
    output_files = ["raster1.tif", "raster2.tif", "raster3.tif", "raster4.tif"]
    nodata_values = [255, 254, 253, 252]  # Different nodata for each raster

    for i in range(4):
        data = np.zeros((768, 768), dtype=np.uint8)

        # Set the center tile to nodata
        data[256:512, 256:512] = nodata_values[i]

        # Set the designated corner to 1
        corner = corner_tiles[i]
        data[corner["row"][0] : corner["row"][1], corner["col"][0] : corner["col"][1]] = 1

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(
            output_files[i],
            768,
            768,
            1,
            gdal.GDT_Byte,
            options=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"],
        )

        ds.SetGeoTransform(geotransform)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).Fill(0)
        ds.GetRasterBand(1).WriteArray(data)
        ds.GetRasterBand(1).SetNoDataValue(nodata_values[i])
        ds = None

    print("Extent rasters created successfully.")


def create_depth_rasters():
    """
    Create depth rasters (float32) with varying depth values per corner.

    Each raster has valid depth data in one corner (clockwise from top-left):
    - Raster 1: Top-left corner, depth = 1.01
    - Raster 2: Top-right corner, depth = 2.02
    - Raster 3: Bottom-left corner, depth = 3.03
    - Raster 4: Bottom-right corner, depth = 4.04

    The mosaic uses max-merge, so overlapping areas will show the highest depth.
    """
    output_files = ["depth_raster1.tif", "depth_raster2.tif", "depth_raster3.tif", "depth_raster4.tif"]
    nodata = -9999.0

    for i in range(4):
        data = np.zeros((768, 768), dtype=np.float32)

        # Set the center tile to nodata
        data[256:512, 256:512] = nodata

        # Set the designated corner to the depth value
        corner = corner_tiles[i]
        depth_value = 1.01 * (i + 1)
        data[corner["row"][0] : corner["row"][1], corner["col"][0] : corner["col"][1]] = depth_value

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(
            output_files[i],
            768,
            768,
            1,
            gdal.GDT_Float32,
            options=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"],
        )

        ds.SetGeoTransform(geotransform)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).Fill(0)
        ds.GetRasterBand(1).WriteArray(data)
        ds.GetRasterBand(1).SetNoDataValue(nodata)
        ds = None

    print("Depth rasters created successfully.")


if __name__ == "__main__":
    create_extent_rasters()
    create_depth_rasters()
    print("All georeferenced datasets created successfully.")
