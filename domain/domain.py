from typing import List
from pydantic import BaseModel


class ChurnRequest(BaseModel):
    frequency: int
    monetary: int
    avg_order_value: int


class ChurnResponse(BaseModel):
    churn: int


class BatchChurnRequest(BaseModel):
    customers: List[ChurnRequest]


class BatchChurnResponse(BaseModel):
    predictions: List[ChurnResponse]