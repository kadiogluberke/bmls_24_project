import os
import logging
import pandas as pd
from model_pipeline.data_collector import DataCollector
from backend.api_models import PredictionRequest


class FeatureExtractor:
    def __init__(self, zones_filename: str, folder: str, data_collector: DataCollector):
        self.logger = logging.getLogger(__name__)
        self.zones_filename = zones_filename
        self.folder = folder
        self.data_collector = data_collector
        self.zones_filepath = os.path.join(folder, zones_filename)

        self.df_zone = None
        self._ensure_zones_data()

    def _ensure_zones_data(self):
        if not os.path.exists(self.zones_filepath):
            self.logger.info(f"{self.zones_filename} not found. Downloading...")
            self.df_zone = self.data_collector.collect_zones_data()
            self.data_collector.store_zones_data_to_csv(
                self.df_zone, self.zones_filename, self.folder
            )
        else:
            self.df_zone = pd.read_csv(self.zones_filepath)

    def _get_dummy_columns(self):
        test_df_top = pd.read_csv("data/test.csv", nrows=0)
        test_df_top.drop(columns=["trip_time"], inplace=True)
        return test_df_top.columns.tolist()

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_columns = self._get_dummy_columns()

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        return df[expected_columns]

    def _merge_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.merge(
            df, self.df_zone, left_on="PULocationID", right_on="LocationID", how="left"
        )
        df = df.rename(
            columns={
                "Borough": "pickup_borough",
                "Zone": "pickup_zone",
                "service_zone": "pickup_service_zone",
            }
        )
        df = df.drop(columns=["PULocationID", "pickup_zone", "LocationID"])

        df = pd.merge(
            df, self.df_zone, left_on="DOLocationID", right_on="LocationID", how="left"
        )
        df = df.rename(
            columns={
                "Borough": "dropoff_borough",
                "Zone": "dropoff_zone",
                "service_zone": "dropoff_service_zone",
            }
        )
        df = df.drop(columns=["DOLocationID", "dropoff_zone", "LocationID"])

        return df

    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:

        # Manually create one-hot encoding for categorical columns
        pickup_borough = f"pickup_borough_{df.at[0, 'pickup_borough']}"
        pickup_service_zone = f"pickup_service_zone_{df.at[0, 'pickup_service_zone']}"
        dropoff_borough = f"dropoff_borough_{df.at[0, 'dropoff_borough']}"
        dropoff_service_zone = (
            f"dropoff_service_zone_{df.at[0, 'dropoff_service_zone']}"
        )

        # Initialize the dummy columns with zeros
        for column in self._get_dummy_columns():
            if column not in df.columns:
                df[column] = 0

        # Set the respective column to 1
        if pickup_borough in df.columns:
            df[pickup_borough] = 1
        if pickup_service_zone in df.columns:
            df[pickup_service_zone] = 1
        if dropoff_borough in df.columns:
            df[dropoff_borough] = 1
        if dropoff_service_zone in df.columns:
            df[dropoff_service_zone] = 1

        return df

    def extract_features(self, request: PredictionRequest) -> pd.DataFrame:
        self.logger.info("Start feature extraction for inference")

        request_datetime = pd.to_datetime(request.request_datetime)

        features = {
            "trip_distance": request.trip_distance,
            "pickup_hour": request_datetime.hour,
            "pickup_minute": request_datetime.minute,
            "pickup_dayofweek": request_datetime.dayofweek,
            "pickup_dayofmonth": request_datetime.day,
            "is_from_airport": request.Airport,
            "PULocationID": request.PULocationID,
            "DOLocationID": request.DOLocationID,
        }

        df = pd.DataFrame([features])
        df = self._merge_zones(df)
        df = self._one_hot_encode(df)
        df = self._reorder_columns(df)
        self.logger.info("Feature extraction for inference completed")
        return df
