from catboost import CatBoostRegressor
import numpy as np
from src.domain.interfaces import ITrainer


class CatBoostTrainer(ITrainer):
    """
    Реализация ITrainer для обучения моделей CatBoost.
    Оборачивает вызов CatBoostRegressor с заданными параметрами.
    """
    def train(self, x: np.ndarray, y: np.ndarray, params: dict) -> CatBoostRegressor:
        """
        Обучает модель CatBoost на предоставленных данных.

        Параметры
        ----------
        x : np.ndarray
            Матрица признаков формы (n_samples, n_features).
        y : np.ndarray
            Целевая переменная. Может быть одномерной (n_samples,)
            или двумерной (n_samples, n_targets) для multi‑target регрессии.
        params : dict
            Словарь параметров для CatBoostRegressor (например, iterations, depth, learning_rate).
            Должен соответствовать документации CatBoost.

        Возвращает
        -------
        CatBoostRegressor
            Обученная модель CatBoost.

        Исключения
        ----------
        Exception
            Перехватывает и логирует ошибки обучения, после чего пробрасывает исключение дальше.
        """
        try:
            model = CatBoostRegressor(**params, allow_writing_files=False)
            model.fit(x, y, verbose=False)
            return model
        except Exception as e:
            print(f"CatBoost training error: {e}")
            print(f"x shape: {x.shape}, y shape: {y.shape}, x dtype: {x.dtype}")
            raise
