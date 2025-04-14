import data.acquisition
import data.preprocessing
import data.raw_data_generator

acq = data.acquisition.DataAcquisition()
ppr = data.preprocessing.DataPreprocessor() 
rdg = data.raw_data_generator.RawDataGenerator()

def pipeline(regions: list[str], year: int, region: str):
    for region in regions:
        acq.get_data_from_cds(year, region)
        climate_df = rdg.load_from_netcdf(region)
        