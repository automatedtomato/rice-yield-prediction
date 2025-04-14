import cdsapi
import os
from dotenv import load_dotenv

COORDINATES = {
    'narita': [35.54, 140.14, 35.43, 140.28],
    'asahi': [35.78, 140.58, 35.69, 140.75],
    'ichihara': [35.55, 140.03, 35.24, 140.25],
    'katori': [35.95, 140.43, 35.76, 140.64],
    'sanmu': [35.68, 140.34, 35.56, 140.51]
}

load_dotenv()

class DataAcquisition:
    def __init__(self, url: str, key: str):
        self.url = url or os.getenv('CDS_API_URL', 'https://cds.climate.copernicus.eu/api')
        self.key = key or os.getenv('CDS_API_KEY')
        

    def get_data_from_cds(self, year:int, region: str) -> None:    
        dataset = "reanalysis-era5-land-monthly-means"
        request = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": [
                "2m_temperature",
                "soil_temperature_level_1",
                "volumetric_soil_water_layer_1",
                "surface_net_solar_radiation",
                "total_precipitation"
            ],
            "year": [i for i in range(1990, year+1)],
            "month": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12"
            ],
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": COORDINATES[region]
        }

        target = f'../data/raw/era5/era5_{region}.nc'

        client = cdsapi.Client(url=self.url, key=self.key)
        client.retrieve(dataset, request, target)