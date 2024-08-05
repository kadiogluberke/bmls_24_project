from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List
from model_executor import ModelExecutor

router = APIRouter()

_model_executor = None


def get_model_executor() -> ModelExecutor:
    global _model_executor

    if _model_executor is None:
        _model_executor = ModelExecutor(model_path="model.json")

    return _model_executor


class PredictionRequest(BaseModel):
    data: List[List[float]]


class PredictionResponse(BaseModel):
    predictions: List[float]


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model_executor: ModelExecutor = Depends(get_model_executor),
):
    predictions = model_executor.predict(request.data)
    return PredictionResponse(predictions=predictions)
