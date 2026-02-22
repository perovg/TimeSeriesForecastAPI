from src.infrastructure.strategies.direct import DirectForecastStrategy

class MultiOutputForecastStrategy(DirectForecastStrategy):
    """
    Стратегия multi‑target прогнозирования.
    Является наследником DirectForecastStrategy, так как использует тот же подход:
    формирование целевой переменной как вектора будущих значений.
    """
    pass
