import json
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from azureml.core import Model
import numpy as np

def init():
    global model

    # Loading Model Path and model files
    model_root_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model')
    model_json_file = open(os.path.join(model_root_path, "model.json"), 'r')
    model_json = model_json_file.read()

    model_json_file.close()

    model = model_from_json(model_json)

    model.load_weights(os.path.join(model_root_path, 'model.h5'))
    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
 
    
def run(data):
    # De-serializing JSON and converting the image back to a numpy array
    data = json.loads(data)['data']
    data = np.array(data)

    y_hat = np.argmax(model.predict(data), axis=1)
    return y_hat.tolist()