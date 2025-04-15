from datetime import datetime
import json
import os
from typing import Any, Dict, Optional, Tuple

import joblib


class ModelSerializer:
    """Serialize and deserialize models"""
    
    def __init__(self, model_dir: str = "../models"):
        """
        Initialize method

        Args:
            model_dir (str): path to the directory to save the models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def save_model(self,
                    model,
                    features: list,
                    metrics: Dict[str, float],
                    version: Optional[str] = None) -> str:
        """
        Save models

        Args:
            model (object): trained model
            features (list): list of features
            metrics (Dict[str, float]): metrics of the model
            version (str): version of the model

        Returns:
            str: saved version of the model
        """
        
        # If version is not provided, use current timestamp
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        # Create model directory if it doesn't exist
        model_version_dir = os.path.join(self.model_dir, version)
        os.makedirs(model_version_dir, exist_ok=True)
        
        # Save model file
        model_path = os.path.join(model_version_dir, 'model.joblib')
        joblib.dump(model, model_path)
        
        # Save feature info
        feature_path = os.path.join(model_version_dir, 'feature.json')
        with open(feature_path, 'w') as f:
            json.dump(features, f)
            
        # Save mete data
        metadata = {
            'version': version,
            'training_date': datetime.now().isoformat(),
            'metrics': metrics,
            'feature_count': len(features)
        }
        
        metadata_path = os.path.join(model_version_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return version
        
    def load_model(self, version: Optional[str] = None) -> Tuple[Any, list, Dict[str, Any]]:
        """
        Load model
        
        Args:
            version: model to load
        
        Returns:
            tuple: (model, list[features], metadata)
        """
        
        # If version not provided, load latest
        if version is None:
            versions = [d for d in os.listdir(self.model_dir)
                        if os.path.isdir(os.path.join(self.model_dir, d))]
            if not versions:
                raise FileNotFoundError('Model not found')
            
            # Load the most recent date version
            version = sorted(versions)[-1]
            
        model_version_dir = os.path.join(self.model_dir, version)
        
        # Load models
        model_path = os.path.join(model_version_dir, 'model.joblib')
        model = joblib.load(model_path)
        
        # Load features
        feature_path = os.path.join(model_version_dir, 'feature.json')
        with open(feature_path, 'r') as f:
            features = json.load(f)
            
        metadata_path = os.path.join(model_version_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return model, features, metadata
        
    def get_model_info(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model information
        
        Args:
            version: model version to get info of
            
        Returns:
            dict: model info
        """
        
        try:
            _, features, metadata = self.load_model(version)
            
            return {
                'model_version': metadata['version'],
                'training_date': metadata['training_date'],
                'features': features,
                'metrics': metadata['metrics']
            }
        except Exception as e:
            return {'error': str(e)}
        
    def list_models(self) -> list:
        """
        Get available model version list
        
        Returns:
            list: version list
        """
        versions = [d for d in os.listdir(self.model_dir)
                    if os.path.isdir(os.path.join(self.model_dir, d))]
        
        return sorted(versions)