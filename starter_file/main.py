from azureml.data.dataset_factory import FileDatasetFactory
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def register_and_download():
    data = FileDatasetFactory.upload_directory("./Covid19-dataset",target=ws.get_default_datastore())    
    data.download(target=".", overwrite=False)

    return data

def get_data():

    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
                './train',
                target_size=(150,150),
                batch_size=32)