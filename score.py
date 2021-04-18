import json
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from azureml.core import Model
import numpy as np

def init():
    global model

    print("Files at os.getenv('AZUREML_MODEL_DIR')/model:")
    print(os.listdir(os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model')))
    model_root_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model') #Model.get_model_path('covid-pneumonia-cnn')
    
    model_json_file = open(os.path.join(model_root_path, "model.json"), 'r')
    model_json = model_json_file.read()

    model_json_file.close()

    model = model_from_json(model_json)

    model.load_weights(os.path.join(model_root_path, 'model.h5'))
    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
 
    
def run(data):
    # data = np.array(json.loads(raw_data)['data'])
    data = json.loads(data)['data']
    data = np.array(data)
    # pil_img = image.array_to_img(data)

    y_hat = np.argmax(model.predict(data), axis=1)
    return y_hat.tolist()