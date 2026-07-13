import pandas as pd
import joblib

from domain.domain import (
    ChurnRequest,
    ChurnResponse
)


class ChurnService:

    def __init__(self):
        self.path_to_model = "artifacts/churn_model_lr.pkl"
        self.path_scaler = "artifacts/StandardScaler.pkl"

        self.model = self.load_artifact(self.path_to_model)
        self.scaler = self.load_artifact(self.path_scaler)

    def load_artifact(self, path_to_artifact):
        """Load model/scaler artifact."""
        with open(path_to_artifact, "rb") as f:
            artifact = joblib.load(f)

        return artifact

    # -----------------------------
    # Single customer preprocessing
    # -----------------------------
    def preprocess(self, request: ChurnRequest) -> pd.DataFrame:

        data = pd.DataFrame([{
            "frequency": request.frequency,
            "monetary": request.monetary,
            "avg_order_value": request.avg_order_value
        }])

        data_scaled = self.scaler.transform(data)

        return pd.DataFrame(
            data_scaled,
            columns=data.columns
        )

    # -----------------------------
    # Batch preprocessing
    # -----------------------------
    def preprocess_batch(
        self,
        requests: list[ChurnRequest]
    ) -> pd.DataFrame:

        data = pd.DataFrame([
            {
                "frequency": r.frequency,
                "monetary": r.monetary,
                "avg_order_value": r.avg_order_value
            }
            for r in requests
        ])

        data_scaled = self.scaler.transform(data)

        return pd.DataFrame(
            data_scaled,
            columns=data.columns
        )

    # -----------------------------
    # Single prediction
    # -----------------------------
    def predict_churn(
        self,
        request: ChurnRequest
    ) -> ChurnResponse:

        data = self.preprocess(request)

        prediction = self.model.predict(data)[0]

        return ChurnResponse(
            churn=int(prediction)
        )

    # -----------------------------
    # Batch prediction
    # -----------------------------
    def predict_churn_batch(
        self,
        requests: list[ChurnRequest]
    ) -> list[ChurnResponse]:

        data = self.preprocess_batch(requests)

        predictions = self.model.predict(data)

        return [
            ChurnResponse(churn=int(pred))
            for pred in predictions
        ]