import os
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta


class DataCollector:
    def __init__(self):
        load_dotenv()

        self.db_username = os.getenv("DB_USERNAME")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_database = os.getenv("DB_DATABASE")

        self.connection_string = f"postgresql://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"

        self.engine = create_engine(self.connection_string)

    def get_data(self, days: int = 30) -> pd.DataFrame:
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

                # Concatenate all data into a single DataFrame
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    return combined_df
                else:
                    return pd.DataFrame()
        except SQLAlchemyError as e:
            print(f"Error occurred: {e}")
            return pd.DataFrame()
        
    def store_data()
