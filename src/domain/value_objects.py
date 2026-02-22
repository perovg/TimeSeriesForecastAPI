from dataclasses import dataclass
from datetime import datetime
from typing import Dict

@dataclass(frozen=True)
class TimePoint:
    """Одно наблюдение временного ряда"""
    timestamp: datetime
    endogenous: float
    exogenous: Dict[str, float]  # предполагаем, что признаки уже предобработаны

@dataclass(frozen=True)
class ForecastHorizon:
    """Горизонт предсказания"""
    value: int

    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("Horizon must be positive")

@dataclass(frozen=True)
class LagCount:
    """Количество лагов, используемых для предсказания"""
    value: int

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Lags cannot be negative")

@dataclass(frozen=True)
class MetricName:
    """Название метрики, которая будет оценивать качество модели после обучения"""
    name: str  # можно позже добавить Enum
