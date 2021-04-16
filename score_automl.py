import json
import numpy as np
import os
from sklearn.externals import joblib
from azureml.core import Model


def init():
    print("Started init")
    global model
    print("Getting Model Path")
    model_path = Model.get_model_path('covid-pneumonia-automl', _workspace=ws)#'./outputs/model.pkl')
    print("Loading model using joblib")
    model = joblib.load(model_path)
    print("Finished init")

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error