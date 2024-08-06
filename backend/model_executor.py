import xgboost as xgb
from typing import List
import pandas as pd


class ModelExecutor:
    def __init__(self, model_path: str):
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict(self, df: pd.DataFrame) -> float:
        dmatrix = xgb.DMatrix(df)
        predictions = self.model.predict(dmatrix)
        return predictions[0]
