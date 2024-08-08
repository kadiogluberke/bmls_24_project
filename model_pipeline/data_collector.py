import os
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
import logging


class DataCollector:
    def __init__(self):
        load_dotenv()
        self.logger = logging.getLogger(__name__)

        self.db_username = os.getenv("DB_USERNAME")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_database = os.getenv("DB_DATABASE")

        self.connection_string = f"postgresql://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"

        self.engine = create_engine(self.connection_string)

    def get_data(self, days: int = 10) -> pd.DataFrame:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            with self.engine.connect() as connection:
                inspector = inspect(connection)
                tables = inspector.get_table_names(schema="public")

                all_data = []
                for i in range(days + 1):
                    date = start_date + timedelta(days=i)
                    table_name = f"trips_{date.strftime('%Y_%m_%d')}"
                    if table_name in tables:
                        query = text(
                            f"SELECT * FROM {table_name} ORDER BY tpep_pickup_datetime"
                        )
                        result = connection.execute(query)
                        df = pd.DataFrame(result.fetchall(), columns=result.keys())
                        all_data.append(df)

                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    return combined_df
                else:
                    return pd.DataFrame()
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get data: {e}")
            return pd.DataFrame()

    def store_data_to_csv(
        self, data: pd.DataFrame, file_name: str = "data.csv", folder: str = "data"
    ) -> None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path = os.path.join(folder, file_name)
        data.to_csv(file_path, index=False)
        self.logger.info(f"Data stored to {file_path}, {len(data)} rows")

    def collect_zones_data(self) -> pd.DataFrame:
        try:
            with self.engine.connect() as connection:
                query = text("SELECT * FROM taxi_zone_lookup")
                result = connection.execute(query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                self.logger.info(f"Zones data collected, {len(df)} rows")
                return df
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get zones data: {e}")
            return pd.DataFrame()

    def store_zones_data_to_csv(
        self, data: pd.DataFrame, file_name: str = "zones.csv", folder: str = "data"
    ) -> None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_path = os.path.join(folder, file_name)
        data.to_csv(file_path, index=False)
        self.logger.info(f"Zones data stored to {file_path}, {len(data)} rows")

    def run(
        self,
        days: int = 10,
        file_name: str = "data.csv",
        folder: str = "data",
        zones_file_name: str = "zones.csv",
    ) -> None:
        data = self.get_data(days=days)
        if not data.empty:
            self.store_data_to_csv(data, file_name, folder)
        else:
            self.logger.warning("No data to store")

        zones_data = self.collect_zones_data()
        if not zones_data.empty:
            self.store_zones_data_to_csv(zones_data, zones_file_name, folder)
        else:
            self.logger.warning("No zones data to store")
