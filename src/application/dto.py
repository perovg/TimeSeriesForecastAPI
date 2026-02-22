from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime

@dataclass
class TimePointDTO:
    timestamp: datetime
    endogenous: float
    exogenous: Dict[str, float]

@dataclass
class FitModelRequest:
    time_series_id: str
    points: List[TimePointDTO]
    horizon: int
    strategy: str  # 'direct', 'recursive', 'multioutput'
    lags: int
    catboost_params: Dict[str, Any]
    metrics: List[str]

@dataclass
class FitModelResponse:
    model_id: str
    model_base64: str
    metrics: Dict[str, float]
