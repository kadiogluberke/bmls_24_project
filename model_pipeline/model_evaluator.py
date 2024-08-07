import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import logging
import json
import os
from dvclive import Live


class ModelEvaluator:
    def __init__(
        self,
        model_path: str,
        test_file: str,
        report_file: str = "evaluation.json",
        report_folder: str = "reports",
    ):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.test_file = test_file
        # self.report_file = report_file
        # self.report_folder = report_folder
        self.model = None
        self.X_test = None
        self.y_test = None

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            self.logger.info(f"Model loaded from {self.model_path}")
        else:
            self.logger.error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def load_test_data(self):
        test_df = pd.read_csv(self.test_file)
        self.X_test = test_df.drop(columns=["trip_time"])
        self.y_test = test_df["trip_time"]

    def evaluate(self):
        if self.model is None or self.X_test is None:
            self.logger.error("Model or test data is not loaded.")
            return

        test_dmatrix = xgb.DMatrix(self.X_test, label=self.y_test)

        preds = self.model.predict(test_dmatrix)

        rmse = mean_squared_error(self.y_test, preds, squared=False)
        mae = mean_absolute_error(self.y_test, preds)
        mape = mean_absolute_percentage_error(self.y_test, preds)

        self.logger.info(f"RMSE: {rmse}")
        self.logger.info(f"MAE: {mae}")
        self.logger.info(f"MAPE: {mape}")

        with Live(resume=True) as live:
            live.log_metric("RMSE", rmse)
            live.log_metric("MAE", mae)
            live.log_metric("MAPE", mape)

        self.logger.info(f"Evaluation metrics saved")

        # evaluation_report = {
        #     "Test RMSE": rmse,
        #     "Test MAE": mae,
        #     "Test MAPE": mape,
        # }

        return

    # def save_evaluation_report(self, evaluation_report: dict):
    #
    #     if not os.path.exists(self.report_folder):
    #         os.makedirs(self.report_folder)
    #
    #     report_path = os.path.join(self.report_folder, self.report_file)
    #
    #     with open(report_path, "w") as f:
    #         json.dump(evaluation_report, f)
    #
    #     self.logger.info(f"Evaluation report saved to {report_path}")

    def run(self):
        self.load_model()
        self.load_test_data()
        self.evaluate()
        # self.save_evaluation_report(evaluation_report)
