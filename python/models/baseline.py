"""
baseline.py - baseline model for predicting yield

This module offers a baseline model for predicting rice yield
using simple linear regression various features such as 
avg. air temp, avg. soil temp and total precipitation of the growing season
"""

import logging
import os
from typing import Tuple
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

from utils.constants import RAW_DATA_DIR,  REGIONS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class BaselineModel:
    """Baseline model (linear regression)"""
    
    def __init__(self, data_dir: str = RAW_DATA_DIR, output_dir: str = './models/baseline'):
        """
        Initialize method

        Args:
            data_dir (str): path to the directory containing the raw data
            output_dir (str): path to the directory containing the output files
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.regions = REGIONS
        self.stages = {
            'preparation': [1,2,3,4],
            'planting': [5],
            'growing': [6,7],
            'heading': [8,9],
            'harvesting': [10]
        }
        self.model=None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self, filename='combined_data.csv') -> Tuple[dict, dict]:
        """
        Load data from csv file

        Args:
            filename (str): name of the file to load

        Returns:
            dict: dictionary of climate data and yield data
        """
        
        climate_data = {}
        yield_data = {}
        
        for region in self.regions:
            try:
                # Load climate data
                climate_file = os.path.join(self.data_dir, f"{region}_climate_df.csv")
                climate_data[region] = pd.read_csv(climate_file)
                
                # Extract and transform date
                climate_data[region]['date'] = pd.to_datetime(climate_data[region]['valid_time'])
                climate_data[region]['year'] = climate_data[region]['valid_time'].dt.year
                climate_data[region]['month'] = climate_data[region]['valid_time'].dt.month
                
                # Load yield data
                yield_file = os.path.join(self.data_dir, f"{region}_yields_df.csv")
                yield_data[region] = pd.read_csv(yield_file)
                
                logger.info(f'{region} - climate data span: {climate_data[region]["year"].min()} to {climate_data[region]["year"].max()}')
                logger.info(f'{region} - yield data span: {yield_data[region]["Year"].min()} to {yield_data[region]["Year"].max()}')
                logger.info(f'{region} - number of records: {len(yield_data[region])}')
                
            except Exception as e:
                logger.error("Error occurred while loading data for %s: %s", region, str(e))
                
        logger.info("Data loaded")
        return climate_data, yield_data
    
    def aggregate_climate_by_stage(self, climate_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate climate data by stage

        Args:
            climate_df (pd.DataFrame): dataframe of climate data

        Returns:
            pd.DataFrame: dataframes aggregated by stage
        """
        
        result = pd.DataFrame()
        
        # Aggregate climate data by stage
        for stage, months in self.stages.items():
            # Filter data by month
            stage_data = climate_df[climate_df['month']].isin(months)
            
            # Aggregate by year
            agg_data = stage_data.groupby('year').agg({
                'temp_2m': 'mean',
                'soil_temp_l1': 'mean',
                'soil_water_vol_l1': 'mean',
                'net_solar_radiation': 'sum',
                'total_rain': 'sum' 
            }).reset_index()

            # Change column names
            agg_data.columns = ['year'] + [f'{stage}_{col}' for col in agg_data.columns[1:]]
            
            if result.empty:
                result = agg_data
            else:
                result = pd.merge(result, agg_data, on='year')
        return result
    
    def prepare_features(self, climate_data: dict, yield_data: dict) -> pd.DataFrame:
        """
        Prepare features for linear regression, create joined dataframe
        
        Args:
            climate_data (dict): dictionary of climate data
            yield_data (dict): dictionary of yield data 
            
        Returns:
            pd.DataFrame: joined dataframe
        """
        logger.info('Preparing features')
        
        combined_data = []
        
        for region in self.regions:
            try:
                # Aggregate climate data by stage
                climate_df = self.aggregate_climate_by_stage(climate_data[region])
                
                # Join climate and yield data
                joined_df = pd.merge(climate_df, yield_data[region][['Year', 'Yields']], left_on='year', right_on='Year')
                
                # Add region column
                joined_df['City'] = region
                
                combined_data.append(joined_df)
                logger.info(f'Completed feature preparation for {region}')
                
            except Exception as e:
                logger.error("Error occurred while preparing features for %s: %s", region, str(e))
                
        all_data = pd.concat(combined_data, ignore_index=True)
        logger.info(f'Data size of joined data: {all_data.shape}')
        
        return all_data
    
    def train_model(self, data: pd.DataFrame, test_size=0.2, random_state=42) -> LinearRegression:
        """
        Train linear regression model

        Args:
            data (pd.DataFrame): dataframe of joined data
            test_size (float): ratio of test data
            random_state (int): random seed
            
        Returns:
            LinearRegression: trained model
        """
        logger.info('Started training model')
        
        # Select features and target
        self.feature_cols = [
            'growing_temp_2m',
            'growing_soil_temp_l1',
            'growing_total_rain'
        ]
        
        # Transfer regions to dummy variables
        data_encoded = pd.get_dummies(data, columns=['City'], drop_first=True)
        
        X = data_encoded[self.feature_cols + [col for col in data_encoded.columns if 'region_' in col]]
        y = data_encoded['Yields'] 
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        # Show importance of features
        feature_importance = dict(zip(X.columns, self.model.coef_))
        for feature, importance in sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True):
            logger.info(f'{feature}: coef = {importance:.4f}')
            
        logger.info(f'intercept = {self.model.intercept_:.4f}')
        return self.model
    
    def evaluate_model(self) -> dict | None:
        """
        Evaluate model performance
        
        Returns:
            ditc: dictionary of evaluation metrics
        """
        logger.info('Evaluate model performance')
        
        if self.model is None or self.X_train is None or self.y_train is None:
            logger.error('Model is not trained yet, or train data is not loaded')
            return None
        
        y_train_pred = self.model.predict(self.X_train)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        if self.X_test is None or self.y_test is None:
            logger.error('Test data is not loaded')
            return None
            
        y_test_pred = self.model.predict(self.X_test)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        metrics = {
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        }
        
        logger.info(f'Train MSE: {train_mse:.4f}')
        logger.info(f'Train RMSE: {train_rmse:.4f}')
        logger.info(f'Train R^2: {train_r2:.4f}')
        
        logger.info(f'Test MSE: {test_mse:.4f}')
        logger.info(f'Test RMSE: {test_rmse:.4f}')
        logger.info(f'Test R^2: {test_r2:.4f}')
        
        return metrics
    
    def save_model(self, model_file='baseline_model.joblib', feature_file='baseline_features.csv'):
        """
        Save model and features

        Args:
            model_file (str): name of the file to save model
            feature_file (str): name of the file to save features
        """
        logger.info('Saving model and features')
        
        model_path = os.path.join(self.output_dir, model_file)
        joblib.dump(self.model, model_path)
        
        features_path = os.path.join(self.output_dir, feature_file)
        if self.X_train is not None:
            pd.Series(self.X_train.columns).to_csv(features_path, index=False)
        else:
            logger.error('Train data is not loaded')
            return
        
        logger.info(f'Model saved to {model_path}')
        logger.info(f'Features saved to {features_path}')
        
    def baseline_model_pipeline(self) -> dict | None:
        """
        Pipeline for training and evaluating model
        
        Returns:
            dict: metrics
        """
        logger.info('Started pipeline')
        
        try:
            
            # 1. Load data
            climate_data, yield_data = self.load_data()
            
            # 2. Prepare features
            all_data = self.prepare_features(climate_data, yield_data)
            
            # 3. Train model
            self.train_model(all_data)
            
            # 4. Evaluate model
            metrics = self.evaluate_model()
            
            # 5. Save model and features
            self.save_model()
            
            logger.info('Completed pipeline')
            return metrics
        
        except Exception as e:
            logger.error(f"パイプライン実行中にエラーが発生: {str(e)}")
            return None
        
if __name__ == '__main__':
    # Run model building
    model = BaselineModel()
    metrics = model.baseline_model_pipeline()
    
    if metrics:
        print('\n=== Evaluation Metrics ===')
        print(f'Train RMSE: {metrics["train_rmse"]:.3f} kg/ha')
        print(f'Train R^2: {metrics["train_r2"]:.3f}')
        print(f'Test RMSE: {metrics["test_rmse"]:.3f} kg/ha')
        print(f'Test R^2: {metrics["test_r2"]:.3f}')