from azureml.data.dataset_factory import FileDatasetFactory, TabularDatasetFactory
from azureml.core.run import Run
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing import image_dataset_from_directory

# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 
# from keras.callbacks import Callback
# from keras.preprocessing import image_dataset_from_directory

import argparse
import os
import numpy as np
import pandas as pd
       

class LogCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        run.log("accuracy", np.float(logs['acc']))
        # print("\nLogging Accuracy", np.float(logs['accuracy']))        

def convert_to_table(image_data_path):
    table = image_dataset_from_directory(image_data_path,
                                         batch_size=16, 
                                         image_size=(data_width, data_height),
                                         label_mode='int', 
                                         color_mode='grayscale')
    np_it = table.as_numpy_iterator()
    all_images_np = np.empty((0,(data_height * data_width + 1)))

    for element in np_it:
        print("image shape:", element[0].shape)
        flattened_image = np.reshape(element[0],(element[0].shape[0],-1))
        reshaped_labels = np.expand_dims(element[1], 1)
        image_label_np = np.append(flattened_image, reshaped_labels, axis=1)
        all_images_np = np.append(all_images_np, image_label_np, axis=0)

    print("flattened shape:",all_images_np.shape)
    column_names = list(range(all_images_np.shape[1]-1))
    column_names.append('class')
    df = pd.DataFrame(all_images_np, columns=column_names)

    return df

def register_image_data_as_file(ws, local_data_path, dataset_name):
    image_file_data = FileDatasetFactory.upload_directory(local_data_path, target=ws.get_default_datastore())    
    image_file_data.register(ws, dataset_name)
    #image_file_data.download(target_path="./downloaded_data", overwrite=False)

    return data

def register_image_data_as_table(ws, image_data_path, dataset_name):
    image_df = convert_to_table(image_data_path)
    data_factory = TabularDatasetFactory.register_pandas_dataframe(image_df, target=ws.get_default_datastore(), name=dataset_name)
    return data_factory

def get_data():

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
                './Covid19-dataset/train',
                target_size=(data_width, data_height),
                batch_size=32,
                color_mode='grayscale')
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
                './Covid19-dataset/test',
                target_size=(data_width, data_height),
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
    model.add(Dense(args.dense_units, activation='relu'))
    model.add(Dense(3, activation='softmax'))

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

    # print("Number of filters in Conv1 and Conv2:", np.int(args.filter1))
    # print("Number of filters in Conv3:", np.int(args.filter2))
    # print("Number of filters in Conv4:", np.int(args.filter3))
    # print("Number of units in the Dense layer:", np.int(args.dense_units))
    # print("Dropout Rate:", np.float(args.dropout_rate))
    # print("Number of Epochs:", np.int(args.epochs))

    train_gen, test_gen = get_data()
    # plot_data(gen)

    model = getCNNModel(args)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x=train_gen, epochs=args.epochs, verbose=2, callbacks=[LogCallback()])#, workers=6, use_multiprocessing=True)
    results = model.evaluate(x=test_gen, verbose=2)# workers=6, use_multiprocessing=True)

    run.log("accuracy", np.float(results[1]))
    # print("accuracy", np.float(results['accuracy']))   

    os.makedirs('./outputs', exist_ok=True)
    model.save('./outputs/model') 

run = Run.get_context()

# height and width of the image to resize to
data_height = 150
data_width = int(150*1.3)
# register_data()

if __name__ == '__main__':
    main()
    # convert_to_table('./Covid19-dataset/train')