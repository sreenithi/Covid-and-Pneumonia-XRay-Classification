from azureml.data.dataset_factory import FileDatasetFactory
from azureml.core.run import Run
# import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 
import argparse
import os
import numpy as np

# def register_data():
#     data = FileDatasetFactory.upload_directory("./Covid19-dataset/",target=ws.get_default_datastore())    
#     print("type:",type(data))
    
#     data.register(ws, "covid-pneumonia-dataset")
#     #data.download(target_path="./downloaded_data", overwrite=False)

#     return data       

def get_data():

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
                './Covid19-dataset/train',
                target_size=(data_height, data_width),
                batch_size=32,
                color_mode='grayscale')
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
                './Covid19-dataset/test',
                target_size=(data_height, data_width),
                batch_size=4,
                color_mode='grayscale')

    return train_gen, test_gen


def plot_data(image_generator):
    
    batch = next(image_generator)[0]
    
    num_images = batch.shape[0]
    cols = min(5,num_images)
    
    fig, ax = plt.subplots(nrows=1, ncols=cols)

    for i in range(cols):
        image = batch[i]
        ax[i].imshow(image)
        ax[i].axis('off')

def getCNNModel(args):

    model = Sequential()

    model.add(Conv2D(args.filter1, (3,3), input_shape=(data_height, data_width, 1), activation='relu', name='conv1'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(args.filter1, (3,3), activation='relu', name='conv2'))
    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(args.dropout_rate))

    model.add(Conv2D(args.filter2, (3,3), activation='relu', name='conv3'))
    model.add(MaxPooling2D((3,3)))

    model.add(Conv2D(args.filter3, (3,3), activation='relu', name='conv4'))
    model.add(MaxPooling2D((3,3)))

    model.add(Dropout(args.dropout_rate))

    model.add(Flatten())
    model.add(Dense(args.dense_units, activation='softmax'))

    return model

def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--filter1', type=int, default=96, help="Number of filters for the first 2 convolutional layers")
    arg_parser.add_argument('--filter2', type=int, default=64, help="Number of filters for the 3rd convolutional layer")
    arg_parser.add_argument('--filter3', type=int, default=32, help="Number of filters for the 4th convolutional layer")
    arg_parser.add_argument('--dense_units', type=int, default=128, help="Number of units in the dense layer")
    arg_parser.add_argument('--dropout_rate', type=float, default=0.2, help="dropout rate for the 2 dropout layers")
    arg_parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train")
    
    args = arg_parser.parse_args()

    run.log("Number of filters in Conv1 and Conv2:", np.int(args.filter1))
    run.log("Number of filters in Conv3:", np.int(args.filter2))
    run.log("Number of filters in Conv4:", np.int(args.filter3))
    run.log("Number of units in the Dense layer:", np.int(args.dense_units))
    run.log("Dropout Rate:", np.float(args.dropout_rate))
    run.log("Number of Epochs:", np.int(args.epochs))

    train_gen, test_gen = get_data()
    # plot_data(gen)

    model = getCNNModel(args)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x=train_gen, epochs=args.epochs)#, workers=6, use_multiprocessing=True)
    results = model.evaluate(x=test_gen, return_dict=True)# workers=6, use_multiprocessing=True)

    run.log("accuracy", np.float(results['accuracy']))   

    os.makedirs('./outputs', exist_ok=True)
    model.save('./outputs/model') 

run = Run.get_context()

# height and width of the image to resize to
data_height = 256
data_width = 256
# register_data()

if __name__ == '__main__':
    main()