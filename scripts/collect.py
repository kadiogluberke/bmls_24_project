from model_pipeline.data_collector import DataCollector

# Example usage:
if __name__ == "__main__":
    collector = DataCollector()
    data = collector.get_data(days=10)
    print(data.shape)
    print(data.head())
