import numpy as np
from src.domain.interfaces import IMetricCalculator

class MAECalculator(IMetricCalculator):
    """Средняя абсолютная ошибка (Mean Absolute Error)."""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

class RMSECalculator(IMetricCalculator):
    """Среднеквадратичная ошибка (Root Mean Squared Error)."""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

class MSECalculator(IMetricCalculator):
    """Среднеквадратичная ошибка (Mean Squared Error)."""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

class MAPECalculator(IMetricCalculator):
    """Средняя абсолютная процентная ошибка (Mean Absolute Percentage Error)."""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Избегаем деления на ноль: добавляем эпсилон или игнорируем нулевые значения
        mask = y_true != 0
        if not np.any(mask):
            return np.nan
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100

class SMAPECalculator(IMetricCalculator):
    """Симметричная средняя абсолютная процентная ошибка (Symmetric Mean Absolute Percentage Error)."""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        # Избегаем деления на ноль
        mask = denominator != 0
        if not np.any(mask):
            return np.nan
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])) * 100

class R2Calculator(IMetricCalculator):
    """Коэффициент детерминации R²."""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0  # все значения одинаковы
        return float(1 - ss_res / ss_tot)

class MaxErrorCalculator(IMetricCalculator):
    """Максимальная абсолютная ошибка."""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.max(np.abs(y_true - y_pred)))
