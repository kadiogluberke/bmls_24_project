from model_pipeline.data_processor import DataProcessor
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    processor = DataProcessor(
        data_filename="data/data.csv",
        zones_filename="data/zones.csv",
        output_folder="data",
    )
    processor.run()
