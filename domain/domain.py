from pydantic import BaseModel

class ChurnRequest(BaseModel):
    frequency: int
    monetary: int
    avg_order_value: int

class ChurnResponse(BaseModel):
    churn: int