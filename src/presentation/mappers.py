from src.presentation.schemas import FitRequestSchema
from src.application.dto import FitModelRequest, TimePointDTO

def map_request_schema_to_dto(schema: FitRequestSchema) -> FitModelRequest:
    points = [
        TimePointDTO(
            timestamp=p.timestamp,
            endogenous=p.endogenous,
            exogenous=p.exogenous
        )
        for p in schema.points
    ]
    return FitModelRequest(
        time_series_id=schema.time_series_id,
        points=points,
        horizon=schema.horizon,
        strategy=schema.strategy,
        lags=schema.lags,
        catboost_params=schema.catboost_params,
        metrics=schema.metrics,
    )
