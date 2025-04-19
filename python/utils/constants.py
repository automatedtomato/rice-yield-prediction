import os
from pathlib import Path

# Statistics indicators
REGIONS = ["asahi", "ichihara", "katori", "narita", "sanmu"]
REGIONS_KANJI = ['旭市', '市原市', '香取市', '成田市', '山武市']
# REGIONS_DICT = {
#      1: '旭市',
#      2: '市原市',
#      3: '香取市',
#      4: '成田市',
#      5: '山武市'}



MUNICIPALITY_CODES = {
    "asahi": 1013,
    "ichihara": 1017,
    "katori": 1034,
    "narita": 1010,
    "sanmu": 1035,
}

COORDINATES = {
    "narita": [35.54, 140.14, 35.43, 140.28],
    "asahi": [35.78, 140.58, 35.69, 140.75],
    "ichihara": [35.55, 140.03, 35.24, 140.25],
    "katori": [35.95, 140.43, 35.76, 140.64],
    "sanmu": [35.68, 140.34, 35.56, 140.51],
}


# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")