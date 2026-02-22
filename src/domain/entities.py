from typing import List
from .value_objects import TimePoint
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timezone

def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

class TimeSeries:
    """Класс для временного ряда"""
    def __init__(self, points: List[TimePoint], series_id: Optional[str] = None):
        if not points:
            raise ValueError("Time series must contain at least one point")
        # Проверка уникальности временных меток
        timestamps = [p.timestamp for p in points]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError("Duplicate timestamps in time series")
        self.points = sorted(points, key=lambda p: p.timestamp)  # всегда сортируем по времени
        self.series_id = series_id

    def __len__(self) -> int:
        return len(self.points)

    def get_endogenous_array(self):
        """Возвращает список значений эндогенного признака временного ряда."""
        return [p.endogenous for p in self.points]

    def get_exogenous_data(self):
        """Возвращает список словарей значений экзогенных признаков временного ряда."""
        return [p.exogenous for p in self.points]


def utc_now() -> datetime:
    """Возвращает текущее время в часовом поясе UTC без timezone."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

@dataclass
class TrainedModel:
    """Сущность обученной модели с метаданными."""
    model_id: str
    model_data: bytes
    series_id: str
    horizon: int
    lags: int
    strategy: str
    metrics: Dict[str, float]
    created_at: datetime = field(default_factory=utc_now)
    metadata: Optional[Dict[str, Any]] = None
