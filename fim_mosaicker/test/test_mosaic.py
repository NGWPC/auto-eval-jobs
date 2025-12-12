import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import rasterio

# Resolve paths for tests
TEST_DIR = Path(__file__).parent.resolve()
MOCK_DATA_DIR = TEST_DIR / "mock_data"
PROJECT_ROOT = TEST_DIR.parent  # Assumes test is in fim_mosaicker/test/
SCRIPT_PATH = PROJECT_ROOT / "mosaic.py"


class TestMosaicScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_data_dir = MOCK_DATA_DIR
        os.makedirs(cls.mock_data_dir, exist_ok=True)

        cls.script_path = SCRIPT_PATH
        if not cls.script_path.exists():
            raise FileNotFoundError(f"Script not found at {cls.script_path}")

        cls.raster_paths = [
            cls.mock_data_dir / "raster1.tif",
            cls.mock_data_dir / "raster2.tif",
            cls.mock_data_dir / "raster3.tif",
            cls.mock_data_dir / "raster4.tif",
        ]
        # Convert paths to strings for JSON serialization
        cls.raster_paths_str_list = [str(p) for p in cls.raster_paths]

        cls.output_path = cls.mock_data_dir / "mosaicked_raster.tif"

        # Check for input files
        missing_files = [p for p in cls.raster_paths if not p.exists()]
        if missing_files:
            print(f"\nWARNING: The following test raster files are missing: {missing_files}")
            print("Run 'make_test_mosaic_data.py' in the mock_data directory or ensure they exist.")
            print("Tests requiring these files may fail or be skipped.")
            cls.input_files_exist = False
        else:
            cls.input_files_exist = True

    def test_mosaic_creation_extent(self):
        """Tests mosaic creation with fim_type='extent'."""
        if not self.input_files_exist:
            self.skipTest("Skipping test because input raster files are missing")

        if self.output_path.exists():
            self.output_path.unlink()

        # Prepare raster_paths as a single space-separated string
        raster_paths_space_str = " ".join(self.raster_paths_str_list)

        cmd = [
            sys.executable,  # Use the current Python interpreter
            str(self.script_path),
            "--raster_paths",
            raster_paths_space_str,  # Pass paths as space-separated list
            "--mosaic_output_path",
            str(self.output_path),
            "--fim_type",
            "extent",
            "--parallel_blocks",
            "1",  # Explicitly test sequential mode
        ]

        # --- Set Environment Variables for subprocess ---
        test_env = os.environ.copy()
        # Set GDAL_CACHEMAX as per job conventions / example.env
        test_env["GDAL_CACHEMAX"] = "1024"  # Example value from example.env
        # Set log level if needed for debugging test failures
        # test_env['LOG_LEVEL'] = 'DEBUG'

        print(f"\nRunning command: {' '.join(cmd)}")
        print(f"With Environment: GDAL_CACHEMAX={test_env['GDAL_CACHEMAX']}")

        # Run with full output capture
        result = subprocess.run(cmd, capture_output=True, text=True, env=test_env)

        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")  # This should contain JSON logs

        self.assertEqual(
            result.returncode,
            0,
            f"Script failed with return code {result.returncode}. Check STDERR logs above.",
        )

        self.assertTrue(
            self.output_path.exists(),
            f"Mosaic output file was not created at {self.output_path}",
        )

        try:
            with rasterio.open(self.output_path) as src:
                # Check data type (uint8 for extent)
                self.assertEqual(
                    src.dtypes[0],
                    "uint8",
                    f"Expected uint8 data type for extent, got {src.dtypes[0]}",
                )

                # Check nodata value (255 for extent)
                self.assertEqual(
                    src.nodata,
                    255,
                    f"Expected 255 nodata value for extent, got {src.nodata}",
                )

                # Check that data exists
                data = src.read(1)
                self.assertTrue(
                    np.any(data != src.nodata),
                    "Raster contains no valid data (all nodata values)",
                )

                # For extent type, check that all valid values are either 0 or 1
                valid_data_mask = data != src.nodata
                if np.any(valid_data_mask):  # Only check if there is valid data
                    valid_data = data[valid_data_mask]
                    unique_values = np.unique(valid_data)
                    is_valid = np.all((unique_values == 0) | (unique_values == 1))
                    self.assertTrue(
                        is_valid,
                        f"Extent raster contains values other than 0 or 1 (and nodata). Found: {unique_values}",
                    )

                # Check that there are 1s in the output (mock data ensures corners = 1)
                self.assertTrue(
                    np.any(data == 1),
                    "Extent raster doesn't contain any 1 values, but should have corner tiles with 1s based on mock data.",
                )

        except rasterio.RasterioIOError as e:
            self.fail(f"Failed to open or read the output raster file: {self.output_path}. Error: {e}")

    def test_mosaic_creation_depth(self):
        """Tests mosaic creation with fim_type='depth'."""
        # Use depth-specific raster files
        depth_raster_paths = [
            self.mock_data_dir / "depth_raster1.tif",
            self.mock_data_dir / "depth_raster2.tif",
            self.mock_data_dir / "depth_raster3.tif",
            self.mock_data_dir / "depth_raster4.tif",
        ]
        missing_depth_files = [p for p in depth_raster_paths if not p.exists()]
        if missing_depth_files:
            self.skipTest(f"Skipping test because depth raster files are missing: {missing_depth_files}")

        depth_output_path = self.mock_data_dir / "mosaicked_depth_raster.tif"
        if depth_output_path.exists():
            depth_output_path.unlink()

        # Prepare raster_paths as a single space-separated string
        raster_paths_space_str = " ".join(str(p) for p in depth_raster_paths)

        cmd = [
            sys.executable,
            str(self.script_path),
            "--raster_paths",
            raster_paths_space_str,
            "--mosaic_output_path",
            str(depth_output_path),
            "--fim_type",
            "depth",
            "--parallel_blocks",
            "1",
        ]

        test_env = os.environ.copy()
        test_env["GDAL_CACHEMAX"] = "1024"

        print(f"\nRunning command: {' '.join(cmd)}")
        print(f"With Environment: GDAL_CACHEMAX={test_env['GDAL_CACHEMAX']}")

        result = subprocess.run(cmd, capture_output=True, text=True, env=test_env)

        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

        self.assertEqual(
            result.returncode,
            0,
            f"Script failed with return code {result.returncode}. Check STDERR logs above.",
        )

        self.assertTrue(
            depth_output_path.exists(),
            f"Mosaic output file was not created at {depth_output_path}",
        )

        try:
            with rasterio.open(depth_output_path) as src:
                # Check data type (float32 for depth)
                self.assertEqual(
                    src.dtypes[0],
                    "float32",
                    f"Expected float32 data type for depth, got {src.dtypes[0]}",
                )

                # Check nodata value (-9999 for depth)
                self.assertEqual(
                    src.nodata,
                    -9999,
                    f"Expected -9999 nodata value for depth, got {src.nodata}",
                )

                # Check that data exists
                data = src.read(1)
                self.assertTrue(
                    np.any(data != src.nodata),
                    "Raster contains no valid data (all nodata values)",
                )

                # For depth type, check that valid values are non-negative floats
                valid_data_mask = data != src.nodata
                if np.any(valid_data_mask):
                    valid_data = data[valid_data_mask]
                    # Depth values should be >= 0 (no negative depths)
                    self.assertTrue(
                        np.all(valid_data >= 0),
                        f"Depth raster contains negative values: min={valid_data.min()}",
                    )

                # Check that expected depth values are present in the mosaic
                # Mock data has corners with depths: 1.01, 2.02, 3.03, 4.04
                expected_depths = [1.01, 2.02, 3.03, 4.04]
                for expected in expected_depths:
                    self.assertTrue(
                        np.any(np.isclose(data, expected, atol=0.01)),
                        f"Depth raster doesn't contain expected depth value {expected}",
                    )

        except rasterio.RasterioIOError as e:
            self.fail(f"Failed to open or read the output raster file: {depth_output_path}. Error: {e}")


if __name__ == "__main__":
    unittest.main()
