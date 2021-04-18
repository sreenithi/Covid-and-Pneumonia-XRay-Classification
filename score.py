import json
import os
from keras.models import model_from_json
from azure.core.model import Model
import numpy as np

def init():
    global model

    model_root_path = os.getenv('AZUREML_MODEL_DIR') #Model.get_model_path('covid-pneumonia-cnn')
    
    model_json_file = open(os.path.join(model_root_path, "model.json"), 'r')
    model_json = model_json_file.read()

    model_json_file.close()

    model = model_from_json(model_json)

    model.load_weights(os.path.join(model_root_path, 'model.h5'))
    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
 
    
def run(data_gen):
    # data = np.array(json.loads(raw_data)['data'])
    y_hat = np.argmax(model.predict(data_gen), axis=1)
    return y_hat.tolist()