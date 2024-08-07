import prefect
import pandas as pd
from model_pipeline.data_collector import DataCollector
from model_pipeline.data_processor import DataProcessor
from model_pipeline.model_trainer import ModelTrainer
from model_pipeline.model_evaluator import ModelEvaluator


@prefect.task
def data_collection():
    collector = DataCollector()
    collector.run()


@prefect.task
def data_processing():
    processor = DataProcessor("data/data.csv", "data/zones.csv", "data")
    processor.run()


@prefect.task
def model_training(n_trials: int):
    trainer = ModelTrainer("data/train.csv", "data/val.csv", "data/test.csv")
    trainer.run(n_trials=n_trials)


@prefect.task
def model_evaluator():
    evaluator = ModelEvaluator(model_path="models/xgb.json", test_file="data/test.csv")
    evaluator.run()


def generate_flow_run_name():
    time_now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return f"{time_now}"


@prefect.flow(
    name="NY Taxi - Model Prefect Pipeline", flow_run_name=generate_flow_run_name
)
def model_pipeline(n_trials: int = 50) -> None:
    data_collection()
    data_processing()
    model_training(n_trials=n_trials)
    model_evaluator()


if __name__ == "__main__":
    model_pipeline.serve(name="NY Taxi - Scheduled Pipeline", cron="*/4 * * * *")
