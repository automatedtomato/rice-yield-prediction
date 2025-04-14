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

REGION = ['asahi', 'ichihara', 'katori', 'narita', 'sanmu']

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Load and preprocess data"""
        
    def __init__(self, data_dir: str='../data/raw', nc_dir: str='../data/raw/era5'):
        """
        Args:
            data_dir (str): path to the directory containing the raw data
            nc_dir (str): path to the directory containing the netcdf data
        """
        self.data_dir = data_dir
        self.nc_dir = nc_dir
        self.climate_data = {}
        self.yields_data = {}
        self.combined_data = None
        
    def load_data(self, use_netcdf: bool = False) -> None:
        """
        Load data of all areas
        
        Args:
            use_netcdf (bool): if True, load netcdf data; otherwise, load csv
        
        """
        
        rdg  = RawDataGenerator()

        logger.info(f'Loading data from {self.data_dir}')
        
        for region in REGION:
            try:
                if use_netcdf:
                    self.climate_data[region] = rdg.load_from_netcdf(region)
                    logger.info(f'Loaded NetCDF climate data for {region}')
                
                else:
                    # load from csv
                    climate_path = os.path.join(self.data_dir, f'{region}_climate.csv')
                    if os.path.exists(climate_path):
                        self.climate_data[region] = self._load_climate_data(climate_path)
                        logger.info(f'Loaded climate data for {region}')
                    else:
                        logger.warning(f'Climate data for {region} not found: {climate_path}')
                    
                    yields_path = os.path.join(self.data_dir, f'{region}_yields.csv')
                    if os.path.exists(yields_path):
                        self.yields_data[region] = self._load_yield_data(yields_path)
                        logger.info(f'Loaded yields data for {region}')
                    else:
                        logger.warning(f'Yields data for {region} not found: {yields_path}')
                    
            except Exception as e:
                logger.error('Failed to load data for region %s: %s', region, str(e))
        
        logger.info('Data loading complete')
    
    
        
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
        
        # TODO: add more preprocessing
            
        return df