from urllib import request, response

import pandas as pd

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from domain.domain import ChurnRequest, ChurnResponse

class ChurnService():
    def __init__(self):
        self.path_to_model = 'artifacts/churn_model_lr.pkl'
        self.path_scaler = 'artifacts/StandardScaler.pkl'
        self.model = self.load_artifact (self.path_to_model)
        self.scaler = self.load_artifact (self.path_scaler)

    def load_artifact(self, path_to_artifact):
        '''Load from the specified path.'''
        with open(path_to_artifact, 'rb') as f:
            artifact = joblib.load(f)
        return artifact   

    def preprocess(self, request: ChurnRequest) -> pd.DataFrame:
        data_dict = {
            'frequency': request.frequency,
            'monetary': request.monetary,
            'avg_order_value': request.avg_order_value
        }

        data = pd.DataFrame([data_dict])

        data_scaled = self.scaler.transform(data)

        data = pd.DataFrame(
            data_scaled,
            columns=data.columns
        )

        return data
    
    def predict_churn(self, request: ChurnRequest) -> ChurnResponse:
        # Convert the request to a DataFrame
        data = self.preprocess(request)
        
        # Predict churn
        churn_prediction = self.model.predict(data)[0]
        
        response = ChurnResponse(churn=int(churn_prediction))
        return response

#test the service    
#if __name__ == "__main__":
 #   service = ChurnService()
  #  request = ChurnRequest(frequency=5, monetary=100, avg_order_value=20)
   # response = service.predict_churn(request)
    #print(response)
    
    
    
#run: "python -m service.churn_service" before running the above code, make sure to have the model and scaler artifacts in the specified paths.