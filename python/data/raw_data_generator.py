"""
raw_data_generator.py - dataframe/csv generation pipeline

This module offers a pipeline for dataframe generation from:
- raw netcdf data to dataframe
- raw csv data to prepared csv
"""

import logging
import os
import pandas as pd
import xarray as xr

from utils.constants import RAW_DATA_DIR, REGIONS

logger = logging.getLogger(__name__)


class RawDataGenerator:
    def __init__(self, raw_dir: str = RAW_DATA_DIR, era5_data_dir: str = RAW_DATA_DIR + "/era5"):
        self.raw_dir = raw_dir
        self.era5_data_dir = era5_data_dir

    def load_from_netcdf(self, region: str) -> pd.DataFrame:
        """
        Load netcdf data and transform to dataframe

        Args:
            region (str): name of the region

        Returns:
            pd.DataFrame: dataframe of the netcdf data
        """

        nc_path = os.path.join(self.era5_data_dir, f"era5_{region}.nc")

        if not os.path.exists(nc_path):
            logger.warning(f"NetCDF data for {region} not found: {nc_path}")
            return pd.DataFrame()

        try:
            ds = xr.open_dataset(nc_path)
            df = ds.to_dataframe()

            df.insert(0, "city", f"{region.capitalize()}")
            df.drop(columns=["number", "expver"], inplace=True)
            df.rename(
                columns={
                    "t2m": "temp_2m",
                    "stl1": "soil_temp_l1",
                    "swvl1": "soil_water_vol_l1",
                    "ssr": "net_solar_radiation",
                    "tp": "total_rain",
                },
                inplace=True,
            )
            df = df.groupby(["valid_time", "city"], as_index=False)[
                [
                    "temp_2m",
                    "soil_temp_l1",
                    "soil_water_vol_l1",
                    "net_solar_radiation",
                    "total_rain",
                ]
            ].mean()
            df = df.sort_values(by=["city", "valid_time"], ascending=[True, False])
            df.reset_index(drop=True, inplace=True)

            ds.close()

            logger.info(
                f"Transformed NetCDF data for {region} to dataframe successfully"
            )
            return df

        except Exception as e:
            logger.error(
                "Failed to transform NetCDF data for region %s: %s", region, str(e)
            )
            return pd.DataFrame()

    def load_from_stats(self, year: int) -> None:
        """
        Load raw yield data and transform to prepared csv

        Args:
            year (int): year of the data
        """
        df = pd.read_csv(os.path.join(self.raw_dir, f"chiba_yields_{year}.csv"))

        cols_idx = [0, 7]
        cols_to_keep = [df.columns[i] for i in cols_idx]
        df = df[cols_to_keep]

        df.rename(
            columns={df.columns[0]: "City", df.columns[1]: "Yields"}, inplace=True
        )

        df.insert(0, "Year", year)
        df.insert(1, "CityId", df.City.apply(self.city_to_id))
        df.City = df.City.apply(self.id_to_city)

        for city in REGIONS:
            existing_city_df = pd.read_csv(
                os.path.join(self.raw_dir, f"{city}_yields_df.csv")
            )
            city_df = df.loc[df.City] == city

            new_df = pd.concat([existing_city_df, city_df])

            df = df.sort_values(by=["CityId", "Year"], ascending=[True, False])
            df.reset_index(drop=True, inplace=True)

            new_df.to_csv(
                os.path.join(self.raw_dir, f"{city}_yields_df.csv"), index=False
            )

    def city_to_id(self, city_name: list[str]):
        """
        Helper function to assign city id

        Args:
            city_name (list[str]): list of city names

        Returns:
            int or None: city id
        """
        city_id_map = {"成田市": 1, "旭市": 2, "市原市": 3, "香取市": 4, "山武市": 5}

        for city, city_id in city_id_map.items():
            if city in city_name:
                return city_id
        return None

    def id_to_city(self, city_id: int):
        """
        Helper function to assign city name

        Args:
            city_id (int): city id

        Returns:
            str or None: city name
        """
        city_id_map = {1: "成田市", 2: "旭市", 3: "市原市", 4: "香取市", 5: "山武市"}

        for id in city_id_map:
            if city_id == id:
                return city_id_map[id]
        return None
