from fastapi import FastAPI

from service.churn_service import ChurnService

from domain.domain import ChurnRequest, ChurnResponse

churn_app=FastAPI()

@churn_app.post("/predict_churn")
async def predict_churn(request: ChurnRequest)->ChurnResponse:
    service = ChurnService()
    response = service.predict_churn(request)
    return response