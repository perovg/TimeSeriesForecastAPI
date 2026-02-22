from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from .entities import TimeSeries
from .value_objects import ForecastHorizon, LagCount

class IForecastStrategy(ABC):
    """Стратегия подготовки данных для обучения и прогнозирования."""
    @abstractmethod
    def prepare_train_data(
        self, series: TimeSeries, horizon: ForecastHorizon, lags: LagCount
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает x (признаки) и y (цель) для обучения."""
        pass

    @abstractmethod
    def forecast(
        self, model: Any, series: TimeSeries, horizon: ForecastHorizon, lags: LagCount
    ) -> np.ndarray:
        """Выполняет прогноз на последние horizon точек ряда."""
        pass

    @abstractmethod
    def extract_test_values(
        self, series: TimeSeries, horizon: ForecastHorizon
    ) -> np.ndarray:
        """Извлекает истинные значения тестового периода (последние horizon точек)."""
        pass

class ITrainer(ABC):
    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> Any:
        """Обучает модель и возвращает объект модели (например, CatBoost)."""
        pass

class IMetricCalculator(ABC):
    """Вычисляет метрику качества прогноза по истинным и предсказанным значениям."""
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class IModelRepository(ABC):
    """Отвечает за сохранение и загрузку обученных моделей вместе с метаданными."""
    @abstractmethod
    def save(self, model_id: str, model_data: bytes, metadata: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def load(self, model_id: str) -> Tuple[bytes, Dict[str, Any]]:
        pass


class StrategyFactory(Dict[str, Any]):
    """Словарь со стратегиями прогнозирования."""
    pass

class MetricFactory(Dict[str, Any]):
    """Словарь с калькуляторами метрик."""
    pass
