from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List, Dict, Any

class TimePointSchema(BaseModel):
    timestamp: datetime
    endogenous: float
    exogenous: Dict[str, float]

class FitRequestSchema(BaseModel):
    time_series_id: str
    points: List[TimePointSchema] = Field(..., min_items=1)
    horizon: int = Field(..., gt=0)
    strategy: str = Field(..., pattern="^(direct|recursive|multioutput)$")
    lags: int = Field(..., ge=0)
    catboost_params: Dict[str, Any] = Field(default_factory=dict)
    metrics: List[str] = Field(..., min_items=1)

    @classmethod
    @field_validator('points')
    def unique_timestamps(cls, points: List[TimePointSchema]) -> List[TimePointSchema]:
        timestamps = [p.timestamp for p in points]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError('timestamps must be unique')
        return points

class FitResponseSchema(BaseModel):
    model_id: str
    model_base64: str
    metrics: Dict[str, float]
