import json
import numpy as np
import os
# from sklearn.externals import joblib
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
    # print("Loading model using pickle")
    # model = pickle.load(model_path)
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