import uuid
import base64
import pickle
from src.domain import TimeSeries, TimePoint
from src.domain import ForecastHorizon, LagCount
from src.domain import ITrainer, IModelRepository, StrategyFactory, MetricFactory
from src.application import FitModelRequest, FitModelResponse


class FitModelUseCase:
    """
    Сценарий использования для обучения модели CatBoost на временном ряде.

    Координирует процесс:
      - создания доменного объекта временного ряда,
      - выбора стратегии прогнозирования,
      - подготовки данных,
      - обучения модели,
      - вычисления метрик на тестовом периоде,
      - сериализации и сохранения модели.
    Зависимости (стратегии, тренер, репозиторий, фабрика метрик) внедряются через конструктор.
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        trainer: ITrainer,
        model_repo: IModelRepository,
        metric_factory: MetricFactory,
    ):
        self.strategy_factory = strategy_factory
        self.trainer = trainer
        self.model_repo = model_repo
        self.metric_factory = metric_factory

    def execute(self, request: FitModelRequest) -> FitModelResponse:
        """
        Выполняет полный цикл обучения модели по запросу.

        Параметры
        ----------
        request : FitModelRequest
            DTO с данными временного ряда, параметрами обучения и списком метрик.

        Возвращает
        -------
        FitModelResponse
            DTO с идентификатором модели, её представлением в base64 и вычисленными метриками.

        Исключения
        ----------
        ValueError
            Если запрошенная стратегия или метрика не зарегистрированы,
            или если данные не проходят валидацию в стратегии.
        """
        points = [TimePoint(p.timestamp, p.endogenous, p.exogenous) for p in request.points]
        series = TimeSeries(points, request.time_series_id)

        strategy = self.strategy_factory.get(request.strategy)
        if not strategy:
            raise ValueError(f"Unknown strategy: {request.strategy}")

        horizon = ForecastHorizon(request.horizon)
        lags = LagCount(request.lags)

        if request.strategy in ("direct", "multioutput"):
            if "loss_function" not in request.catboost_params:
                request.catboost_params["loss_function"] = "MultiRMSE"

        x_train, y_train = strategy.prepare_train_data(series, horizon, lags)
        model = self.trainer.train(x_train, y_train, request.catboost_params)

        y_true = strategy.extract_test_values(series, horizon)
        y_pred = strategy.forecast(model, series, horizon, lags)

        metrics = {}
        for metric_name in request.metrics:
            calculator = self.metric_factory.get(metric_name)
            if not calculator:
                raise ValueError(f"Unknown metric: {metric_name}")
            metrics[metric_name] = calculator.calculate(y_true, y_pred)

        model_bytes = pickle.dumps(model)
        model_base64 = base64.b64encode(model_bytes).decode('utf-8')

        model_id = str(uuid.uuid4())
        metadata = {
            "series_id": request.time_series_id,
            "horizon": request.horizon,
            "lags": request.lags,
            "strategy": request.strategy,
            "metrics": metrics,
        }
        self.model_repo.save(model_id, model_bytes, metadata)

        return FitModelResponse(
            model_id=model_id,
            model_base64=model_base64,
            metrics=metrics,
        )
