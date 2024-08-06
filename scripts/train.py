from model_pipeline.model_trainer import ModelTrainer
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    trainer = ModelTrainer("data/train.csv", "data/val.csv", "data/test.csv")
    trainer.run(n_trials=50)
