from fastapi import APIRouter, HTTPException
from dishka import FromDishka
from dishka.integrations.fastapi import inject
from src.presentation.schemas import FitRequestSchema, FitResponseSchema
from src.presentation.mappers import map_request_schema_to_dto
from src.application.use_cases.fit_model import FitModelUseCase
import logging


logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/fit", response_model=FitResponseSchema)
@inject
async def fit_model(
    request: FitRequestSchema,
    use_case: FromDishka[FitModelUseCase],
):
    try:
        dto = map_request_schema_to_dto(request)
        response = use_case.execute(dto)
        return FitResponseSchema(
            model_id=response.model_id,
            model_base64=response.model_base64,
            metrics=response.metrics
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Unhandled exception in /fit")
        raise HTTPException(status_code=500, detail="Internal server error")
