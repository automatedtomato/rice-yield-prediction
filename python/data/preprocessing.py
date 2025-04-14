"""
preprocessing.py - data preprocessing pipeline

This module offers a pipeline for data preprocessing, including:
- data loading
- data cleaning
- data joining
"""

import logging
import os
import pandas as pd

from python.data.raw_data_generator import RawDataGenerator

REGIONS = ["asahi", "ichihara", "katori", "narita", "sanmu"]

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Load and preprocess data"""

    def __init__(self, data_dir: str = "../data/raw", nc_dir: str = "../data/raw/era5"):
        """
        Args:
            data_dir (str): path to the directory containing the raw data
            nc_dir (str): path to the directory containing the netcdf data
        """
        self.data_dir = data_dir
        self.nc_dir = nc_dir
        self.climate_data = {}
        self.yield_data = {}
        self.combined_data = None

    def load_data(self, use_netcdf: bool = False) -> None:
        """
        Load data of all areas

        Args:
            use_netcdf (bool): if True, load netcdf data; otherwise, load csv

        """

        rdg = RawDataGenerator()

        logger.info(f"Loading data from {self.data_dir}")

        for region in REGIONS:
            try:
                if use_netcdf:
                    self.climate_data[region] = rdg.load_from_netcdf(region)
                    logger.info(f"Loaded NetCDF climate data for {region}")

                else:
                    # load from csv
                    climate_path = os.path.join(self.data_dir, f"{region}_climate.csv")
                    if os.path.exists(climate_path):
                        self.climate_data[region] = self._load_climate_data(
                            climate_path
                        )
                        logger.info(f"Loaded climate data for {region}")
                    else:
                        logger.warning(
                            f"Climate data for {region} not found: {climate_path}"
                        )

                    yields_path = os.path.join(self.data_dir, f"{region}_yields.csv")
                    if os.path.exists(yields_path):
                        self.yield_data[region] = self._load_yield_data(yields_path)
                        logger.info(f"Loaded yields data for {region}")
                    else:
                        logger.warning(
                            f"Yields data for {region} not found: {yields_path}"
                        )

            except Exception as e:
                logger.error("Failed to load data for region %s: %s", region, str(e))

        logger.info("Data loading complete")

    def _load_climate_data(self, file_path: str) -> pd.DataFrame:
        """
        Load climate data and perform basic preprocessing

        Args:
            file_path (str): file path to the climate data

        Returns:
            pd.Dataframe: preprocessed climate data frame
        """

        df = pd.read_csv(file_path)

        # TODO: add more preprocessing

        return df

    def _load_yield_data(self, file_path: str) -> pd.DataFrame:
        """
        Load yield data and perform basic preprocessing

        Args:
            file_path (str): file path to the yield data

        Returns:
            pd.Dataframe: preprocessed yield data frame
        """

        df = pd.read_csv(file_path)

        return df

    def clean_data(self) -> None:
        """
        Process outliers, missing values, etc.
        """

        logger.info("Data cleaning started")

        # Clean climate data
        for region, df in self.climate_data.items():
            try:
                # Process missing values

                # e.g. linear interpolation
                # if df.isnull().sum().sum() > 0:
                #     df.interpolate(method='linear', inplace=True)
                #     logger.info(f'Filled missing values in climate data for {region} with linear interpolation')

                # e.g. drop rows with missing values
                # df.dropna(inplace=True)
                # logger.info(f'Dropped missing values in climate data for {region}')

                # e.g. fill with mean
                # df.fillna(df.mean(), inplace=True)
                # logger.info(f'Filled missing values in climate data for {region} with mean value')

                # e.g. fill with median
                # df.fillna(df.median(), inplace=True)
                # logger.info(f'Filled missing values in climate data for {region} with median value')

                # e.g. fill with mode
                # df.fillna(df.mode().iloc[0], inplace=True)
                # logger.info(f'Filled missing values in climate data for {region} with mode value')

                # e.g. fill with zero
                # df.fillna(0, inplace=True)
                # logger.info(f'Filled missing values in climate data for {region} with zero value')

                # e.g. fill with forward fill
                # df.fillna(method='ffill', inplace=True)
                # logger.info(f'Filled missing values in climate data for {region} with forward fill')

                # e.g. fill with backward fill
                # df.fillna(method='bfill', inplace=True)
                # logger.info(f'Filled missing values in climate data for {region} with backward fill')

                # Process outliers
                # e.g. Clip values with greater than 3 times the standard deviation
                for col in [
                    "temp_2m",
                    "soil_temp_l1",
                    "soil_water_vol_l1",
                    "net_solar_radiation",
                    "total_rain",
                ]:
                    if col in df.columns:
                        mean, std = df[col].mean, df[col].std()
                        df[col] = df[col].clip(
                            lower=mean - 3 * std, upper=mean + 3 * std
                        )
                        logger.info(f"Clipped outliers in {col} for {region}")

                # e.g. Drop rows with greater than 3 times the standard deviation
                # for col in ['temp_2m', 'soil_temp_l1', 'soil_water_vol_l1', 'net_solar_radiation', 'total_rain']:
                #     if col in df.columns:
                #         mean, std = df[col].mean, df[col].std()
                #         df = df[df[col].between(mean - 3*std, mean + 3*std)]
                #         logger.info(f'Dropped outliers in {col} for {region}')

                self.climate_data[region] = df

            except Exception as e:
                logger.error(
                    "Error occurred while cleaning data for %s: %s", region, str(e)
                )

        # Clean yield data
        for region, df in self.yield_data.items():
            try:
                # Process missing values

                # e.g. drop rows with missing values
                # df.dropna(inplace=True)
                # logger.info(f'Dropped missing values in yield data for {region}')

                # e.g. fill with mean
                # df.fillna(df.mean(), inplace=True)
                # logger.info(f'Filled missing values in yield data for {region} with mean value')

                # e.g. fill with median
                # df.fillna(df.median(), inplace=True)
                # logger.info(f'Filled missing values in yield data for {region} with median value')

                # e.g. fill with mode
                # df.fillna(df.mode().iloc[0], inplace=True)
                # logger.info(f'Filled missing values in yield data for {region} with mode value')

                # e.g. fill with zero
                # df.fillna(0, inplace=True)
                # logger.info(f'Filled missing values in yield data for {region} with zero value')

                # e.g. fill with forward fill
                # df.fillna(method='ffill', inplace=True)
                # logger.info(f'Filled missing values in yield data for {region} with forward fill')

                # e.g. fill with backward fill
                # df.fillna(method='bfill', inplace=True)
                # logger.info(f'Filled missing values in yield data for {region} with backward fill')

                # e.g. fill with linear interpolation
                # df.interpolate(method='linear', inplace=True)
                # logger.info(f'Filled missing values in yield data for {region} with linear interpolation')

                # Process outliers
                # e.g. Clip values with greater than 3 times the standard deviation
                for col in ["Yields"]:
                    if col in df.columns:
                        mean, std = df[col].mean, df[col].std()
                        df[col] = df[col].clip(
                            lower=mean - 3 * std, upper=mean + 3 * std
                        )
                        logger.info(f"Clipped outliers in {col} for {region}")

                self.yield_data[region] = df

            except Exception as e:
                logger.error(
                    "Error occurred while cleaning data for %s: %s", region, str(e)
                )

        logger.info("Data cleaning complete")

    def join_data(self) -> pd.DataFrame:
        """
        Join climate and yield data into a single dataframe

        Returns:
            pd.DataFrame: joined dataframe
        """

        logger.info("Data joining started")

        joined_df = []

        for region in REGIONS:
            try:
                if region in self.climate_data and region in self.yield_data:
                    # Aggregate yearly climate and yield data for each region
                    climate_df = self.climate_data[region]

                    if "valid_date" in climate_df.columns:
                        climate_df["year"] = climate_df["valid_date"].dt.year

                    yearly_climate = (
                        climate_df.groupby("year")
                        .agg(
                            {
                                "temp_2m": "mean",
                                "soil_temp_l1": "mean",
                                "soil_water_vol_l1": "mean",
                                "net_solar_radiation": "sum",
                                "total_rain": "sum",
                            }
                        )
                        .reset_index()
                    )

                    yield_df = self.yield_data[region]
                    merged_df = pd.merge(
                        yearly_climate, yield_df, on="year", how="inner"
                    )

                    merged_df["city"] = region.capitalize()

                    joined_df.append(merged_df)
                    logger.info(f"Joined data for {region}")
                else:
                    logger.warning(f"Data for {region} is missing")

            except Exception as e:
                logger.error("Error occurred while joining data: %s", str(e))

        if joined_df:
            self.combined_data = pd.concat(joined_df, ignore_index=True)
            logger.info("Data joining complete")
        else:
            logger.warning("No data to join")
            self.combined_data = pd.DataFrame()

        return self.combined_data

    def save_data(
        self, output_path: str = "../../data/processed/combined_data.csv"
    ) -> None:
        """
        Save joined data to csv

        Args:

        """
        if self.combined_data is not None and not self.combined_data.empty:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            self.combined_data.to_csv(output_path, index=False)
            logger.info("Data saved to %s", output_path)
