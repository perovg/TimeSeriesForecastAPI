import numpy as np
from src.domain.interfaces import IForecastStrategy
from src.domain.entities import TimeSeries
from src.domain.value_objects import ForecastHorizon, LagCount

class DirectForecastStrategy(IForecastStrategy):
    """
    Стратегия прямого многошагового прогнозирования (multi‑target).

    Формирует обучающие примеры, где каждый пример состоит из:
      - признаков: lags последних значений эндогенной переменной и экзогенных переменных в текущий момент;
      - целевой переменной: вектор следующих horizon значений эндогенной переменной.

    Прогноз выполняется сразу на весь горизонт с помощью обученной модели,
    которая возвращает вектор длины horizon.
    """
    def prepare_train_data(
        self, series: TimeSeries, horizon: ForecastHorizon, lags: LagCount
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Подготавливает матрицу признаков X и целевую переменную Y для обучения модели.

        Параметры
        ----------
        series : TimeSeries
            Временной ряд с точками (временная метка, эндогенная, экзогенные).
        horizon : ForecastHorizon
            Горизонт прогноза (количество будущих шагов).
        lags : LagCount
            Количество лагов эндогенной переменной, используемых как признаки.

        Возвращает
        -------
        tuple[np.ndarray, np.ndarray]
            X — массив формы (n_samples, n_features), где n_features = lags + число экзогенных переменных.
            Y — массив формы (n_samples, horizon) с целевыми векторами.

        Исключения
        ----------
        ValueError
            Если длина ряда меньше lags + horizon.
        """
        points = series.points
        n = len(points)
        min_required = lags.value + horizon.value
        if n < min_required:
            raise ValueError(f"Not enough points: need {min_required}, have {n}")

        x, y = [], []
        for i in range(lags.value, n - horizon.value + 1):
            # Лаги эндогенной переменной
            lag_values = [points[i - j - 1].endogenous for j in range(lags.value)]
            # Экзогенные признаки в момент i, отсортированные по ключам для стабильности
            exog = points[i].exogenous
            exog_values = [exog[key] for key in sorted(exog.keys())]
            features = lag_values + exog_values
            x.append(features)

            # Целевой вектор — следующие horizon значений
            target = [points[i + k].endogenous for k in range(horizon.value)]
            y.append(target)

        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    def forecast(
        self, model, series: TimeSeries, horizon: ForecastHorizon, lags: LagCount
    ) -> np.ndarray:
        """
        Выполняет прогноз на весь горизонт, используя последние доступные данные.

        Параметры
        ----------
        model : CatBoostRegressor
            Обученная модель, способная возвращать вектор длины horizon.
        series : TimeSeries
            Временной ряд (используются все точки для формирования признаков).
        horizon : ForecastHorizon
            Горизонт прогноза.
        lags : LagCount
            Количество лагов, использованных при обучении.

        Возвращает
        -------
        np.ndarray
            Массив предсказанных значений длины horizon.
        """
        points = series.points
        last_idx = len(points) - 1

        # Формируем вектор признаков из последних lags значений эндогенной переменной
        lag_values = [points[last_idx - j].endogenous for j in range(lags.value)]

        # Экзогенные переменные из последней точки, отсортированные по ключам
        exog = points[last_idx].exogenous
        exog_values = [exog[key] for key in sorted(exog.keys())]

        features = lag_values + exog_values
        x_pred = np.array([features], dtype=np.float32)

        pred = model.predict(x_pred)  # форма (1, horizon)
        return pred.flatten()

    def extract_test_values(
        self, series: TimeSeries, horizon: ForecastHorizon
    ) -> np.ndarray:
        """
        Извлекает истинные значения тестового периода (последние horizon точек ряда).

        Параметры
        ----------
        series : TimeSeries
            Временной ряд.
        horizon : ForecastHorizon
            Горизонт прогноза (количество последних точек для теста).

        Возвращает
        -------
        np.ndarray
            Массив истинных значений длины horizon.
        """
        return np.array([p.endogenous for p in series.points[-horizon.value:]])
