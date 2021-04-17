import json
import numpy as np
import pandas as pd
import os
import joblib
from azureml.core import Model


def init():
    print("\n****************************************")
    print("Started init")
    global model
    print("Getting Model Path")
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model.pkl')
    print("Model Found?:", os.path.isfile(model_path))
    print("Loading model using joblib")
    model = joblib.load(model_path)
    print("Finished init")

def run(data):
    try:
        data_json = json.loads(data)

        print("Converting data to np array")
        data = pd.DataFrame.from_dict(data_json['data'])
        
        print("Predicting data")
        result = model.predict(data)
        
        print("Converting to list")
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
        
    except Exception as e:
        error = str(e)
        return error