import numpy as np
from src.domain.interfaces import IForecastStrategy
from src.domain.entities import TimeSeries
from src.domain.value_objects import ForecastHorizon, LagCount

class RecursiveForecastStrategy(IForecastStrategy):
    """
    Стратегия рекурсивного многошагового прогнозирования.

    Обучает одношаговую модель, которая предсказывает следующее значение
    на основе лагов эндогенной переменной и текущих экзогенных факторов.
    Прогноз на горизонт выполняется итеративно: на каждом шаге полученное
    предсказание добавляется в лаги, а экзогенные переменные берутся
    из последней известной точки (предполагается их неизменность на всём
    горизонте).
    """

    def prepare_train_data(
        self, series: TimeSeries, horizon: ForecastHorizon, lags: LagCount
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Подготавливает обучающие данные для одношаговой модели.

        Из ряда выбираются все возможные окна длины `lags`, для которых
        следующий шаг не выходит за пределы обучающей выборки (т.е.
        исключаются последние `horizon` точек, которые будут использованы
        для тестирования).

        Параметры
        ----------
        series : TimeSeries
            Временной ряд.
        horizon : ForecastHorizon
            Горизонт прогноза (используется только для резервирования
            тестового периода).
        lags : LagCount
            Количество лагов эндогенной переменной.

        Возвращает
        -------
        tuple[np.ndarray, np.ndarray]
            X — матрица признаков формы (n_samples, n_features),
            где n_features = lags + число экзогенных переменных.
            y — одномерный массив целевых значений (следующий шаг).

        Исключения
        ----------
        ValueError
            Если недостаточно данных для формирования хотя бы одного
            обучающего примера (длина ряда меньше или равна `lags`).
        """
        points = series.points
        n = len(points)
        if n <= lags.value:
            raise ValueError(f"Not enough points for training: need > {lags.value}, have {n}")

        x, y = [], []
        # Последний индекс, для которого можно взять y = points[i+1], не заходя в тестовый период
        max_i = n - horizon.value
        for i in range(lags.value, max_i):
            lag_values = [points[i - j - 1].endogenous for j in range(lags.value)]
            exog = points[i].exogenous
            exog_values = [exog[key] for key in sorted(exog.keys())]
            features = lag_values + exog_values
            x.append(features)
            y.append(points[i + 1].endogenous)

        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    def forecast(
        self, model, series: TimeSeries, horizon: ForecastHorizon, lags: LagCount
    ) -> np.ndarray:
        """
        Выполняет рекурсивный прогноз на заданный горизонт.

        Использует обученную одношаговую модель. На каждом шаге:
          - формирует вектор признаков из текущих лагов и последних известных
            экзогенных переменных;
          - получает предсказание;
          - обновляет лаги, добавляя предсказанное значение и удаляя самое старое.

        Параметры
        ----------
        model : CatBoostRegressor
            Обученная одношаговая модель.
        series : TimeSeries
            Временной ряд (используются последние `lags` значений и экзогенные
            переменные из последней точки).
        horizon : ForecastHorizon
            Горизонт прогноза.
        lags : LagCount
            Количество лагов, использованных при обучении.

        Возвращает
        -------
        np.ndarray
            Массив предсказанных значений длины `horizon`.
        """
        points = series.points
        n = len(points)
        # Текущие лаги — последние `lags` значений эндогенной переменной
        current_lags = [points[n - 1 - j].endogenous for j in range(lags.value)]
        # Экзогенные переменные из последней точки (предполагаем их неизменность)
        last_exog = points[-1].exogenous
        exog_values = [last_exog[key] for key in sorted(last_exog.keys())]

        predictions = []
        for _ in range(horizon.value):
            features = current_lags + exog_values
            x_pred = np.array([features], dtype=np.float32)
            pred = model.predict(x_pred)[0]  # ожидается скаляр
            predictions.append(pred)
            # Обновляем лаги: удаляем самый старый, добавляем новое предсказание
            current_lags.pop(0)
            current_lags.append(pred)

        return np.array(predictions)

    def extract_test_values(
        self, series: TimeSeries, horizon: ForecastHorizon
    ) -> np.ndarray:
        """
        Извлекает истинные значения тестового периода (последние `horizon` точек ряда).

        Параметры
        ----------
        series : TimeSeries
            Временной ряд.
        horizon : ForecastHorizon
            Горизонт прогноза.

        Возвращает
        -------
        np.ndarray
            Массив истинных значений длины `horizon`.
        """
        return np.array([p.endogenous for p in series.points[-horizon.value:]])
