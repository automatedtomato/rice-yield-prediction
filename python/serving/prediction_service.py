import sys
import os
import logging

from pure_eval import group_expressions
import grpc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serialization.proto.generated import rice_yield_pb2
from serialization.proto.generated import rice_yield_pb2_grpc

from serialization.model_serializer import ModelSerializer

logger = logging.getLogger(__name__)

class YieldPredictionServicer(rice_yield_pb2_grpc.YieldPredictionServiceServicer):
    """Yield prediction service"""
    
    def __init__(self, model_dir: str='../models'):
        self.serializer = ModelSerializer(model_dir)
        self.default_model, self.features, self.metadata = self.serializer.load_model()
        logger.info(f'Model {self.metadata['version']} loaded from {model_dir}')
        
    def Predict(self, request, context):
        """
        Process prediction request

        Args:
            request : PredictionRequest
            context : gRPC context

        Returns:
            PredictionResponse
        """
        try:
            feature_values = self._extract_features(request)
            
            predicted_yield = float(self.default_model.predict(feature_values)[0])

            feature_contribution = {}
            if hasattr(self.default_model, 'coef_'):
                coeffs = self.default_model.coef_
                intercept = self.default_model.intercept_
                
                for i, feature in enumerate(self.features):
                    feature_contribution[feature] = float(coeffs[i] * feature_values[i])
                
                feature_contribution['intercept']  = float(intercept)
                
            response = rice_yield_pb2.PredictionResponse(
                predicted_yield=predicted_yield,
                feature_contribution=feature_contribution,
                confidence=0.9,  # TODO: Calculate confidence,
                model_info=f'Model version: {self.metadata["version"]}'
            )
            
            logger.info(f'Predicted yield: {predicted_yield} kg/ha')
            
            return response
        
        except Exception  as e:
            logger.error(f'Error during prediction: {str(e)}')
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Prediction error: {str(e)}')
            return rice_yield_pb2.PredictionResponse()
        
    def GetMOdelInfo(self, request, context):
        """
        Get model information

        Args:
            request : ModelInfoRequest
            context : gRPC context

        Returns:
            ModelInfoResponse
        """
        
        try:
            version = request.model.version if request.model.version else None
            model_info = self.serializer.get_model_info(version)
            
            if 'error' in model_info:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(model_info['error'])
                return rice_yield_pb2.ModelInfoResponse()
            
            metrics = model_info['metrics']
            
            response = rice_yield_pb2.ModelInfoResponse(
                version=model_info['version'],
                training_date=model_info['training_date'],
                r_squared=metrics.get('test_r2', 0.0),
                rmse=metrics.get('test_rmse', 0.0),
                features=model_info['features'],
            )
            return response
                
        except Exception as e:
            logger.error(f'Error during getting model info: {str(e)}')
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Get model info error: {str(e)}')
            return rice_yield_pb2.ModelInfoResponse()
        
            
    def _extract_features(self, request):
        """
        Helper method to extract features from the request
        
        Args:
            request : PredictionRequest
            
        Returns:
            feature_values: list
        """
        # Temporary implementation
        
        # for feature in self.features:
            # if feature == 'growing_temp_2m':
            #     feature_values.append(request.temp_2m)
            # elif feature == 'growing_soil_temp_l1':
            #     feature_values.append(request.soil_temp_l1)
            # elif feature == 'growing_total_rain':
            #     feature_values.append(request.total_rain)
            # elif 'region_' in feature:
            #     region_in_feature = feature.split('region_')[1]
            #     feature_values.append(1.0 if region_in_feature == request.region else 0.0)
            # else:
            #     # 該当する特徴量が見つからない場合はゼロで補完
            #     feature_values.append(0.0)
        
        # TODO: Implement feature extraction based on real model
        feature_values=[]
        return feature_values
        
        