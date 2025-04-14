from curses import raw
from data.acquisition import DataAcquisition
from data.preprocessing import DataPreprocessor
from data.raw_data_generator import RawDataGenerator

import os
import logging
from typing import List, Optional, Dict
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

REGIONS = ["asahi", "ichihara", "katori", "narita", "sanmu"]
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")

CDS_API_URL = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.environ.get("CDS_API_KEY", "")
ESTAT_API_KEY = os.environ.get("ESTAT_API_KEY", "")

MUNICIPALITY_CODES = {
    "asahi": 1013,
    "ichihara": 1017,
    "katori": 1034,
    "narita": 1010,
    "sanmu": 1035,
}


class Pipeline:
    def __init__(
        self,
        regions: List[str] = REGIONS,
        start_year: int = 1990,
        end_year: int = 2023,
        raw_dir: str = RAW_DATA_DIR,
        processed_dir: str = PROCESSED_DATA_DIR,
    ):
        """
        Initialize method

        Args:
            regions (list[str]): list of regions
            start_year (int): start year
            end_year (int): end year
            raw_dir (str): path to the directory containing the raw data
            processed_dir (str): path to the directory containing the processed data
        """

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        self.regions = regions
        self.start_year = start_year
        self.end_year = end_year
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

        os.makedirs(os.path.join(self.raw_dir, "era5"), exist_ok=True)
        os.makedirs(os.path.join(self.raw_dir, "estat"), exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        self.acq = DataAcquisition(
            cds_url=CDS_API_URL, cds_key=CDS_API_KEY, estat_key=ESTAT_API_KEY
        )
        self.prep = DataPreprocessor(raw_dir, processed_dir)
        self.gen = RawDataGenerator(raw_dir, processed_dir)

    def run(
        self,
        skip_climate_acquisition: bool = False,
        skip_yield_acquisition: bool = False,
        skip_raw_generation: bool = False,
    ) -> Dict:
        """
        Run pipeline

        Args:
            skip_climate_acquisition (bool): if True, skip climate data acquisition
            skip_crop_acquisition (bool): if True, skip crop data acquisition
            skip_raw_generation (bool): if True, skip raw data generation
        Returns:
            Dict: dictionary of results
        """

        result = {
            "climate_acquisition": {},
            "yield_acquisition": {},
            "raw_generation": {},
            "preprocessing": {},
        }

        # 1. Climate data acquisition

        if not skip_climate_acquisition:
            logger.info("Climate data acquisition started")
            for region in self.regions:
                try:
                    # Get climate data
                    self.acq.get_data_from_cds(self.start_year, self.end_year, region)
                    result["climate_acquisition"][f"{region}_climate"] = "success"

                except Exception as e:
                    logger.error(f"Failed to get climate data for {region}: {e}")
                    result["climate_acquisition"][f"{region}_climate"] = "failed"

            logger.info("Climate data acquisition completed")

        # 2. Yield data acquisition
        if not skip_yield_acquisition:
            logger.info("Yield data acquisition started")

            yield_data_by_region = {region: [] for region in self.regions}

            for year in range(self.start_year, self.end_year + 1):
                try:
                    stats_data_id = self.acq.find_stats_data_id(str(year))

                    if stats_data_id:
                        municipality_code_list = [
                            MUNICIPALITY_CODES[region]
                            for region in self.regions
                            if region in MUNICIPALITY_CODES
                        ]

                        municipality_codes = ",".join(map(str, municipality_code_list))

                        indicator_code = "1026"

                        data_inf = self.acq.get_crop_data(
                            stats_data_id, municipality_codes, indicator_code
                        )

                        if data_inf:
                            for item in data_inf:
                                municipality_code = item.get("@cat01")
                                value = item.get("$")

                                region = next(
                                    (
                                        r
                                        for r, code in MUNICIPALITY_CODES.items()
                                        if code == municipality_code
                                    ),
                                    None,
                                )

                                if region and region in self.regions:
                                    try:
                                        value_float = float(value) if value else None
                                        yield_data_by_region[region].append(
                                            {"year": year, "yield": value_float}
                                        )

                                    except (ValueError, TypeError):
                                        logger.warning(
                                            f"Invalid yield value for {region} in year {year}: {value}"
                                        )

                            result["yield_acquisition"][f"{year}_yield"] = "success"

                        else:
                            logger.warning(f"Failed to get yield data for year {year}")
                            result["yield_acquisition"][f"{year}_yield"] = "no_data"
                    else:
                        logger.warning(f"Failed to get yield data for year {year}")
                        result["yield_acquisition"][f"{year}_yield"] = "no_stats_id"
                except Exception as e:
                    logger.error(f"Failed to get yield data for year {year}: {e}")
                    result["yield_acquisition"][f"{year}_yield"] = f"error: {str(e)}"

            for region, data_list in yield_data_by_region.items():
                if data_list:
                    yield_df = pd.DataFrame(data_list)
                    yield_df.to_csv(
                        os.path.join(self.raw_dir, f"{region}_yields.csv"), index=False
                    )
                    logger.info(f"Yield data for {region} saved")
                else:
                    logger.warning(f"Yield data for {region} not found")
            logger.info("Yield data acquisition completed")
        else:
            logger.info("Yield data acquisition skipped")

        # 3. Raw data generation
        if not skip_raw_generation:
            logger.info("Raw climate data generation started")
            climate_dfs = {}

            for region in self.regions:
                try:
                    climate_df = self.gen.load_from_netcdf(region)

                    if not climate_df.empty:
                        climate_path = os.path.join(
                            self.raw_dir, f"{region}_climate_df.csv"
                        )
                        climate_df.to_csv(climate_path, index=False)
                        climate_dfs[region] = climate_path
                        result["raw_generation"][f"{region}_climate_df"] = climate_path
                    else:
                        logger.warning(f"Could not generate climate data for {region}")
                        result["raw_generation"][f"{region}_climate_df"] = "failed"

                except Exception as e:
                    logger.error(f"Failed to generate climate data for {region}: {e}")
                    result["raw_generation"][
                        f"{region}_climate_df"
                    ] = f"error: {str(e)}"

            logger.info("Raw climate data generation completed")

        else:
            logger.info("Raw climate data generation skipped")

        # 4. Data preprocessing
        logger.info("Data preprocessing started")
        try:
            self.prep.load_data()
            self.prep.clean_data()
            joined_data = self.prep.join_data()

            output_path = os.path.join(self.processed_dir, "combined_data.csv")
            self.prep.save_data(output_path)

            result["preprocessing"] = {
                "status": "success",
                "output_path": output_path,
                "record_count": len(joined_data),
                "columns": (
                    joined_data.columns.tolist() if not joined_data.empty else []
                ),
            }

            logger.info("Data preprocessing completed. Output saved to %s", output_path)
        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            result["preprocessing"] = {
                "status": "error",
                "message": f"Failed to preprocess data: {str(e)}",
            }

        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Pipeline")
    parser.add_argument("--regions", nargs="+", default=REGIONS, help="List of regions")
    parser.add_argument(
        "--start_year", type=int, default=1990, help="Data acquisition start year"
    )
    parser.add_argument(
        "--end_year", type=int, default=2023, help="Data acquisition end year"
    )
    parser.add_argument(
        "--skip_raw_generation", action="store_true", help="Skip raw data generation"
    )
    parser.add_argument(
        "--skip_yield", action="store_true", help="Skip yield data acquisition"
    )
    parser.add_argument(
        "--skip_climate", action="store_true", help="Skip climate data acquisition"
    )

    args = parser.parse_args()

    pipeline = Pipeline(
        regions=args.regions, start_year=args.start_year, end_year=args.end_year
    )

    result = pipeline.run(
        skip_climate_acquisition=args.skip_climate,
        skip_yield_acquisition=args.skip_yield_acquisition,
        skip_raw_generation=args.skip_raw_generation,
    )

    print("\n--- Pipeline Result ---")
    for stage, details in result.items():
        print(f"Stage: {stage.upper()}")
        for key, value in details.items():
            print(f"  {key}: {value}")
