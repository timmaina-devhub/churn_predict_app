from fastapi import FastAPI

from service.churn_service import ChurnService

from domain.domain import (
    ChurnRequest,
    ChurnResponse,
    BatchChurnRequest,
    BatchChurnResponse
)

churn_app = FastAPI()

service = ChurnService()


# ------------------------------------
# Predict one customer
# ------------------------------------
@churn_app.post(
    "/predict_churn",
    response_model=ChurnResponse
)
async def predict_churn(request: ChurnRequest):

    return service.predict_churn(request)


# ------------------------------------
# Predict multiple customers
# ------------------------------------
@churn_app.post(
    "/predict_churn_batch",
    response_model=BatchChurnResponse
)
async def predict_churn_batch(
    request: BatchChurnRequest
):

    predictions = service.predict_churn_batch(
        request.customers
    )

    return BatchChurnResponse(
        predictions=predictions
    )