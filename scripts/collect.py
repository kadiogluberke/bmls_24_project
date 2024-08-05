from model_pipeline.data_collector import DataCollector
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    collector = DataCollector()
    collector.run()
