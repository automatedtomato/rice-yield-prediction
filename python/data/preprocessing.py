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
import numpy as np

from data.raw_data_generator import RawDataGenerator
from utils.constants import REGIONS, RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Load and preprocess data"""

    def __init__(self, raw_dir: str = RAW_DATA_DIR, processed_dir: str = PROCESSED_DATA_DIR):
        """
        Args:
            data_dir (str): path to the directory containing the raw data
            processed_dir (str): path to the directory containing the processed data
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.climate_data = {}
        self.yield_data = {}
        self.combined_data = None

    def load_data(self, use_netcdf: bool = False) -> None:
        """
        Load data of all areas

        Args:
            use_netcdf (bool): if True, load netcdf data; otherwise, load csv

        """

        rdg = RawDataGenerator(self.raw_dir, self.raw_dir + '/era5')

        logger.info(f"Loading data from {self.raw_dir}")

        for region in REGIONS:
            try:
                if use_netcdf:
                    self.climate_data[region] = rdg.load_from_netcdf(region)
                    logger.info(f"Loaded NetCDF climate data for {region}")

                else:
                    # load from csv
                    climate_path = os.path.join(self.raw_dir, f"{region}_climate_df.csv")
                    if os.path.exists(climate_path):
                        self.climate_data[region] = self._load_climate_data(
                            climate_path
                        )
                        logger.info(f"Loaded climate data for {region}")
                    else:
                        logger.warning(
                            f"Climate data for {region} not found: {climate_path}"
                        )

                    yields_path = os.path.join(self.raw_dir, f"{region}_yields_df.csv")
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

    def clean_data(self, imputation_method: str = 'multi_region_trend') -> None:
        """
        Process outliers, missing values, etc.
        
        Args:
            imputation_method (str): method for imputing missing values
            - 'linear': linear interpolation
            - 'spline': spline interpolation
            - 'multi_region_trend': using trends across regions
            - 'domain_knowledge': using domain knowledge with weather data  # TODO: Implement later
        """

        logger.info("Data cleaning started")
        
        # Step 1: Clean outliers in climate data
        self._clean_climate_outliers()
        
        # Step 2: Handle missing years in yield data
        self._create_missing_year_rows()
        
        # Step 3: Impute missing values based on selected method
        if imputation_method == 'linear':
            self._impute_with_linear_interpolation()
        elif imputation_method == 'spline':
            self._impute_with_spline_interpolation()
        elif imputation_method == 'multi_region_trend':
            self._impute_with_multi_region_trend()
        # elif imputation_method == 'domain_knowledge':
        #     self._impute_with_domain_knowledge()
        else:
            logger.warning(f'Unknown imputation method: {imputation_method}. Using multi_region_trend instead.')
            self._impute_with_multi_region_trend()
            
        logger.info("Data cleaning completed")

    def _clean_climate_outliers(self):
        """Clean outliers in climate data"""
        for region, df in self.climate_data.items():
            try:
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

                self.climate_data[region] = df

            except Exception as e:
                logger.error("Error occurred while cleaning data for %s: %s", region, str(e))
                
    def _create_missing_year_rows(self):
        """Create rows for missing years in yield data"""
        
        all_years = set()
        for region, df in self.yield_data.items():
            if 'Year' in df.columns:
                all_years.update(df['Year'].unique())
            elif 'year' in df.columns:
                all_years.update(df['year'].unique())
                
        year_col = 'Year' if 'Year' in next(iter(self.yield_data.values())).columns else 'year'
        all_years = set(sorted(all_years))
        
        # Add missing years to each region's yield data
        for region, df in self.yield_data.items():
            current_years = set(df[year_col].unique())
            missing_years = all_years - current_years
            
            if missing_years:
                logger.info(f"Adding missing years for {region}: {missing_years}")
                
                # Create DataFrame with missing years
                missing_df = pd.DataFrame({year_col: list(missing_years)})
                
                # Add necessary columns with NaN values
                for col in df.columns:
                    if col != year_col:
                        missing_df[col] = float('nan')
                        
                # if region/city column exists, fill it
                city_col = next((col for col in df.columns if col.lower() in ['city', 'region', 'cityid']), None)
                if city_col:
                    missing_df[city_col] = df[city_col].iloc[0]
                    
                # Combine with original data and sort
                self.yield_data[region] = pd.concat([df, missing_df]).sort_values(year_col).reset_index(drop=True)
                
    def _impute_with_linear_interpolation(self) -> None:
        for region, df in self.yield_data.items():
            try:
                yield_col = 'Yields' if 'Yields' in df.columns else 'yields'
                year_col = 'Year' if 'Year' in df.columns else 'year'
                
                # Sort by year
                df = df.sort_values(year_col)
                
                # Apply linear interpolation
                if df[yield_col].isna().any():
                    original_values = df[yield_col].copy()
                    df[yield_col] = df[yield_col].interpolate(method='linear')
                    num_imputed = sum(df[yield_col].notna() & original_values.isna())
                    logger.info(f"Imputed {num_imputed} missing values for {region}")
                    
                self.yield_data[region] = df
                
            except Exception as e:
                logger.error("Error during linear interpolation for %s: %s", region, str(e))
     
    def _impute_with_spline_interpolation(self) -> None:
        """Impute missing values using spline interpolation"""
        for region, df in self.yield_data.items():
            try:
                yield_col = 'Yields' if 'Yields' in df.columns else 'yields'
                year_col = 'Year' if 'Year' in df.columns else 'year'
                
                # Sort by year
                df = df.sort_values(year_col)
                
                # Apply spline interpolation
                if df[yield_col].isna().any():
                    original_values = df[yield_col].copy()
                    df[yield_col] = df[yield_col].interpolate(method='spline', order=3)
                    num_imputed = sum(df[yield_col].notna() & original_values.isna())    
                    logger.info(f"Imputed {num_imputed} missing values for {region}")
                    
                self.yield_data[region] = df
                
            except Exception as e:
                logger.error("Error during spline interpolation for %s: %s", region, str(e))
                
    def _impute_with_multi_region_trend(self):
        """Impute missing values using multi-region trend"""
        try:
            # Collect all yield data
            yield_col = 'Yields' if 'Yields' in next(iter(self.yield_data.values())).columns else 'yields'
            year_col = 'Year' if 'Year' in next(iter(self.yield_data.values())).columns else 'year'
            
            # Create combined DataFrame with all region
            all_yields = []
            for region, df in self.yield_data.items():
                df_copy = df.copy()
                df_copy['region'] = region
                all_yields.append(df_copy)
            
            combined_df = pd.concat(all_yields, ignore_index=True)

            # Create pivot table with year as index and region as columns    
            pivot_df = combined_df.pivot(index=year_col, columns='region', values=yield_col)
            
            # Normalize each region by its mean (where data is available)
            normalized_df = pivot_df.copy()
            for col in normalized_df.columns:
                valid_data = normalized_df[col].dropna()
                if not valid_data.empty:
                    mean_yield = valid_data.mean()
                    normalized_df[col] = normalized_df[col].fillna(mean_yield)
                    
            # For each region with missing data
            for region in normalized_df.columns:
                if normalized_df[region].isna().any():
                    # Get reference regions (those w/o NaN at positions where this region has NaN)
                    missing_idx = normalized_df[region].isna()
                    reference_regions = [r for r in normalized_df.columns if r != region]
                    
                    # Calculate the trend for each reference region
                    if reference_regions:
                        reference_df = normalized_df[reference_regions].copy()
                        reference_trend = reference_df.mean(axis=1)
                        
                        # Calculate scaling factors btw region and reference trend
                        valid_idx = ~normalized_df[region].isna()
                        if valid_idx.any() and not reference_trend.isna().all():
                            scale_factors = normalized_df.loc[valid_idx, region] / reference_trend[valid_idx]
                            avg_scale = scale_factors.mean()
                            
                            # Apply scaling to estimate missing values
                            for year in normalized_df[missing_idx].index:
                                if not np.isnan(reference_trend.loc[year]):
                                    estimated_norm_value = reference_trend[year] * avg_scale
                                    
                                    # Convert back to original yield value
                                    original_mean = pivot_df[region].dropna().mean()
                                    estimated_actual_value = estimated_norm_value * original_mean
                                    
                                    # Updata the original DataFrame
                                    year_idx = self.yield_data[region][year_col] == year
                                    if any(year_idx):
                                        self.yield_data[region].loc[year_idx, yield_col] = estimated_actual_value
                                        logger.info(f'Imputed {region} value for year {year}, using multi-region trend')
                                        
        except Exception as e:
            logger.error("Error during multi-region trend imputation: %s", str(e))
            
    def _impute_with_domain_knowledge(self):
        # TODO: Implement later
        pass
                            
             
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

                    if "valid_time" in climate_df.columns:
                        climate_df["year"] = climate_df["valid_time"].dt.year

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
        self, output_path: str = PROCESSED_DATA_DIR + "combined_data.csv"
    ) -> None:
        """
        Save joined data to csv

        Args:

        """
        if self.combined_data is not None and not self.combined_data.empty:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            self.combined_data.to_csv(output_path, index=False)
            logger.info("Data saved to %s", output_path)
