import sys
import os
import logging
from concurrent import futures
import grpc
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.serialization.proto.generated import rice_yield_pb2_grpc
from serialization.model_serializer import ModelSerializer

from serving.prediction_service import YieldPredictionServicer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

def serve(port: int, model_dir: str):
    """
    Start gRPC server

    Args:
        port (int): port number
        model_dir (str): path to the directory to save the models
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    service = YieldPredictionServicer(model_dir)
    rice_yield_pb2_grpc.add_YieldPredictionServiceServicer_to_server(service, server)
    
    service_address = f'[::]:{port}'
    server.add_insecure_port(service_address)
    server.start()
    
    logger.info(f'Server started on port:{port}')
    logger.info(f'Model directory: {model_dir}')
    
    try:
        while True:
              time.sleep(86400)  
    except KeyboardInterrupt:
        logger.info('Stopping server')
        server.stop(0)
        
if __name__ = '__main__':
    parser = argparse.ArgumentParser(description='Yield prediction gRPC server')
    parser.add_argument('--port', type=int, default=50051, help='Port number (default: 50051)')
    parser.add_argument('--model_dir', type=str, default='../models', help='Model directory (default: ../models)')
    
    args  = parser.parse_args()
    serve(args.port, args.model_dir)