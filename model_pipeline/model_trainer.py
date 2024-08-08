import pandas as pd
import xgboost as xgb
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
import logging
import os
from dvclive import Live


class ModelTrainer:
    def __init__(
        self, train_file: str, val_file: str, test_file: str, model_dir: str = "models"
    ):
        self.logger = logging.getLogger(__name__)
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.model_dir = model_dir
        self.best_model = None

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def load_data(self):
        self.train_df = pd.read_csv(self.train_file)
        self.val_df = pd.read_csv(self.val_file)
        self.test_df = pd.read_csv(self.test_file)

    def separate_features_and_target(self):
        self.X_train = self.train_df.drop(columns=["trip_time"])
        self.y_train = self.train_df["trip_time"]
        self.X_val = self.val_df.drop(columns=["trip_time"])
        self.y_val = self.val_df["trip_time"]
        self.X_test = self.test_df.drop(columns=["trip_time"])
        self.y_test = self.test_df["trip_time"]

    def objective(self, trial: Trial):
        param = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 40, 250),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        }

        train_dmatrix = xgb.DMatrix(self.X_train, label=self.y_train)
        val_dmatrix = xgb.DMatrix(self.X_val, label=self.y_val)

        model = xgb.train(
            param,
            train_dmatrix,
            evals=[(val_dmatrix, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        preds = model.predict(val_dmatrix)
        mae = mean_absolute_error(self.y_val, preds)

        return mae

    def tune_hyperparameters(self, n_trials: int = 100):
        sampler = TPESampler(seed=1)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(self.objective, n_trials=n_trials)

        self.logger.info("Best hyperparameters: ", study.best_params)
        self.best_params = study.best_params

    def train_best_model(self):
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])
        combined_dmatrix = xgb.DMatrix(X_combined, label=y_combined)

        self.best_model = xgb.train(self.best_params, combined_dmatrix)

    def save_model(self, model_filename: str = "xgb.json"):
        model_path = os.path.join(self.model_dir, model_filename)
        self.best_model.save_model(model_path)
        self.logger.info(f"Model saved to {model_path}")
        with Live() as live:
            live.log_artifact(model_path, type="model", name="xgboost")

    def run(self, n_trials: int = 100):
        self.load_data()
        self.separate_features_and_target()
        self.tune_hyperparameters(n_trials)
        self.train_best_model()
        self.save_model()
        with Live(resume=True) as live:
            live.log_param("best_params", self.best_params)
            live.log_param("len_train_data", len(self.y_train))
            live.log_param("len_test_data", len(self.y_test))
