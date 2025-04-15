from tracemalloc import start
import cdsapi
import os
import json
import requests
import logging
from utils.constants import COORDINATES

ESTAT_API_VERSION = "3.0"
STATS_CODE = "00500215"
SURVEY_NAME = "作目統計調査"

logger = logging.getLogger(__name__)

class DataAcquisition:
    def __init__(self, cds_url: str, cds_key: str, estat_key: str):
        self.url = cds_url or os.getenv(
            "CDS_API_URL", "https://cds.climate.copernicus.eu/api"
        )
        self.key = cds_key or os.getenv("CDS_API_KEY")
        self.estat_key = estat_key or os.getenv("ESTAT_API_KEY")

    def get_data_from_cds(self, start_year: int, end_year: int, region: str) -> None:
        dataset = "reanalysis-era5-land-monthly-means"
        request = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": [
                "2m_temperature",
                "soil_temperature_level_1",
                "volumetric_soil_water_layer_1",
                "surface_net_solar_radiation",
                "total_precipitation",
            ],
            "year": [i for i in range(start_year, end_year + 1)],
            "month": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ],
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": COORDINATES[region],
        }

        target = f"../data/raw/era5/era5_{region}.nc"

        client = cdsapi.Client(url=self.url, key=self.key)
        try:
            client.retrieve(dataset, request, target)
        except Exception as e:
            logger.error(f'Failed to get data from CDS: {e}', exc_info=True)

    def find_stats_data_id(self, year_str: str):
        """
        Find stats data id by survey name and year
        Args:
            year_str: 'YYYY'
        """

        endpoint = (
            f"http://api-esta.go.jp/rest{ESTAT_API_VERSION}/app/json/getStatsList"
        )
        params = {
            "appId": self.estat_key,
            "statsCode": STATS_CODE,
            "surveyYears": year_str,
            "searchWord": "水稲 市町村別 千葉県",
            "limit": 20,
        }

        print(f"\n---{year_str}年の統計表ID検索（{SURVEY_NAME} / {STATS_CODE}）---")
        if not self.estat_key or self.estat_key == "":
            print("Error: set app ID")
            return None
        print(f"Request URL (getStatsList): {endpoint}")
        print("Parameters: ")
        masked_params = params.copy()

        if len(self.estat_key) > 8:
            masked_params["appId"] = self.estat_key[:4] + "..." + self.estat_key[-4:]
        else:
            masked_params["appId"] = "********"
        print(json.dumps(masked_params, indent=2, ensure_ascii=False))

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            result = data.get("GET_STATUS_LIST", {}).get("RESULT", {})
            status = result.get("STATUS", -1)

            if status == 0:
                datalist_inf = data.get("GET_STATUS_LIST", {}).get("DATALIST_INF", {})
                table_inf_list = datalist_inf.get("TABLE_INF", [])
                if isinstance(table_inf_list, dict):
                    table_inf_list = [table_inf_list]

                if not table_inf_list:
                    print(
                        f"  -> Result: Could not find any stat data for{year_str}. Please try another year."
                    )
                    return None

                print(
                    f"  -> Result: Found {len(table_inf_list)} stats data for{year_str}."
                )

                # Select most appropriate dataset
                selected_table = None
                for table in table_inf_list:
                    title_obj = table.get("TITLE", {})
                    title = title_obj.get("STATISTICS_NAME")
                    if title and "収穫量" in str(title):
                        selected_table = table
                        print(
                            f"  /> Options: ID={selected_table.get('@id')}, Title={title}"
                        )
                        break

                if not selected_table and table_inf_list:
                    selected_table = table_inf_list[0]
                    title_obj = selected_table.get("TITLE", {})
                    title = (
                        title_obj.get("$")
                        if isinstance(title_obj, dict)
                        else selected_table.get("TITLE")
                    )
                    print(
                        f"  -> Selected (first option): ID={selected_table.get('@id')}, Title={title}"
                    )

                if selected_table:
                    print(f"  -> Selected stats data for {selected_table.get('@id')}")
                    return selected_table.get("@id")
                else:
                    print(
                        "  -> Error: found candidate but could not find stats data ID."
                    )
                    return None

            elif status == 1:
                print(
                    f"  -> Result: could not find stats data for {year_str} (no data)."
                )
                return None
            else:
                error_msg = result.get("ERROR_MSG", "Unknown error")
                print(f"  -> Failed to get stats list (STATUS: {status}): {error_msg}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request Error (getStatsList): {e}")
            return None
        except json.JSONDecodeError:
            print("Failed to decode JSON response (getStatsList).")
            return None
        except Exception as e:
            print(f"Unexpected Error (getStatsList): {e}")
            return None

    def get_crop_data(
        self, stats_data_id: str, municipality_codes: str, indicator_code: str
    ):
        """
        Get data from stats found by find_stats_data_id, municipality_codes and indicator_code
        指定された統計表ID, 市町村コードリスト, 指標コードでデータを取得する。
        cdCat01 が市町村、cdCat02が指標と仮定。
        """
        endpoint = (
            f"https://api.e-stat.go.jp/rest/{ESTAT_API_VERSION}/app/json/getStatsData"
        )
        params = {
            "appId": self.estat_key,
            "statsDataId": stats_data_id,
            "cdCat01": municipality_codes,  # Comma separated
            "cdCat02": indicator_code,
            "lang": "J",
            "metaGetFlg": "Y",
            "cntGetFlg": "N",
            "explanationGetFlg": "N",  # 解説は省略
            "annotationGetFlg": "Y",
            "replaceSpChars": "0",
        }

        print(f"\n--- Get Stats Data (ID: {stats_data_id}) ---")
        print(f"Request URL (getStatsData): {endpoint}")
        print("Parameters:")
        masked_params = params.copy()

        if self.estat_key is not None and len(self.estat_key) > 8:
            masked_params["appId"] = self.estat_key[:4] + "..." + self.estat_key[-4:]
        else:
            masked_params["appId"] = "********"

        if len(masked_params.get("cdCat01", "")) > 50:  # if more than 50, shorten
            masked_params["cdCat01"] = masked_params["cdCat01"][:50] + "..."
        print(json.dumps(masked_params, indent=2, ensure_ascii=False))

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            result = data.get("GET_STATS_DATA", {}).get("RESULT", {})
            status = result.get("STATUS", -1)
            error_msg = result.get("ERROR_MSG", "不明なエラー")

            if status == 0:
                print("  -> Get stats data successfully.")
                statistical_data = data.get("GET_STATS_DATA", {}).get(
                    "STATISTICAL_DATA", {}
                )
                data_inf = statistical_data.get("DATA_INF", {}).get("VALUE", [])
                total_num = statistical_data.get("RESULT_INF", {}).get(
                    "TOTAL_NUMBER", 0
                )
                print(f"  -> Total number: {total_num}")

                # Check data
                print("  -> Example data (first 5):")
                if not data_inf:
                    print("    No data matched.")
                else:
                    if isinstance(data_inf, dict):
                        data_inf = [data_inf]  # 1件の場合
                    for i, item in enumerate(data_inf[:5]):
                        municipality_code = item.get("@cat01")
                        time_code = item.get("@time")
                        value = item.get("$")
                        unit = item.get("@unit")
                        annotation = item.get("@annotation")
                        print(
                            f"    {i+1}: Cities={municipality_code}, Time={time_code}, Value={value} {unit or ''}",
                            end="",
                        )
                        if annotation:
                            print(f" (注釈:{annotation})", end="")
                        print()
                    if len(data_inf) > 5:
                        print("    ...")

                # Check for continuous data
                next_key = statistical_data.get("RESULT_INF", {}).get("NEXT_KEY")
                if next_key:
                    print(f"  -> Continuous data exists (nextKey: {next_key})")
                # IF NEEDED: pagenation

                return data_inf  # 取得したデータリストを返す

            elif status == 1:
                print(f"  -> Could not find data: {error_msg}")
                return None
            else:
                print(f"  -> Failed to get stats data (STATUS: {status}): {error_msg}")
                return None

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error(getStatsData): {e.response.status_code}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error (getStatsData): {e}")
            return None
        except json.JSONDecodeError:
            print("Failed to decode JSON response (getStatsData)")
            return None
        except Exception as e:
            print(f"Unexpected error (getStatsData): {e}")
            return None
