from dishka import Provider, Scope, provide
from src.infrastructure.strategies import (
    DirectForecastStrategy,
    RecursiveForecastStrategy,
    MultiOutputForecastStrategy,
)
from src.infrastructure.ml import CatBoostTrainer
from src.infrastructure.repositories import InMemoryModelRepository
from src.application.use_cases import FitModelUseCase
from src.application.services.metrics import (
    MAECalculator,
    RMSECalculator,
    MSECalculator,
    MAPECalculator,
    SMAPECalculator,
    R2Calculator,
    MaxErrorCalculator,
)
from src.domain import StrategyFactory, MetricFactory

class AppProvider(Provider):
    """Провайдер зависимостей для DI-контейнера Dishka."""
    scope = Scope.REQUEST

    @provide
    def provide_strategy_factory(self) -> StrategyFactory:
        """Предоставляет фабрику стратегий прогнозирования, сопоставляя имя стратегии с её реализацией."""
        return StrategyFactory({
            "direct": DirectForecastStrategy(),
            "multioutput": MultiOutputForecastStrategy(),
            "recursive": RecursiveForecastStrategy()
        })

    @provide
    def provide_metric_factory(self) -> MetricFactory:
        """Предоставляет фабрику калькуляторов метрик, сопоставляя имя метрики с её реализацией."""
        return MetricFactory({
            "mae": MAECalculator(),
            "rmse": RMSECalculator(),
            "mse": MSECalculator(),
            "mape": MAPECalculator(),
            "smape": SMAPECalculator(),
            "r2": R2Calculator(),
            "max_error": MaxErrorCalculator(),
        })

    @provide
    def provide_trainer(self) -> CatBoostTrainer:
        """Предоставляет объект для обучения моделей CatBoost."""
        return CatBoostTrainer()

    @provide
    def provide_repository(self) -> InMemoryModelRepository:
        """Предоставляет репозиторий для сохранения и загрузки обученных моделей."""
        return InMemoryModelRepository()

    @provide
    def provide_use_case(
            self,
            strategy_factory: StrategyFactory,
            trainer: CatBoostTrainer,
            repo: InMemoryModelRepository,
            metric_factory: MetricFactory,
    ) -> FitModelUseCase:
        """Создаёт и предоставляет сценарий использования для обучения модели."""
        return FitModelUseCase(
            strategy_factory=strategy_factory,
            trainer=trainer,
            model_repo=repo,
            metric_factory=metric_factory
        )
