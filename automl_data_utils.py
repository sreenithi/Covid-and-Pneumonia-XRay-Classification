from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.run import Run
import tensorflow as tf
# from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from keras.preprocessing import image_dataset_from_directory


import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
               

def convert_to_table(image_data_path):

    train_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(
                image_data_path,
                target_size=(data_width, data_height),
                batch_size=16,
                class_mode="sparse",
                color_mode='grayscale')

    all_images_np = np.empty((0,(data_height * data_width + 1)))

    num_batches = round(train_gen.samples / train_gen.batch_size)

    count = 0
    for element in train_gen:
        flattened_image = np.reshape(element[0],(element[0].shape[0],-1))
        flattened_image /= 255.0
        reshaped_labels = np.expand_dims(element[1], 1)
        image_label_np = np.append(flattened_image, reshaped_labels, axis=1)
        all_images_np = np.append(all_images_np, image_label_np, axis=0)

        count += 1
        if count == num_batches:
            break

    column_names = list(range(all_images_np.shape[1]-1))
    column_names.append('class')
    df = pd.DataFrame(all_images_np, columns=column_names)
    df['class'] = df['class'].astype(int)

    return df    



# def convert_to_table(image_data_path):
#     table = image_dataset_from_directory(image_data_path,
#                                          batch_size=16, 
#                                          image_size=(data_width, data_height),
#                                          label_mode='int', 
#                                          color_mode='grayscale')
#     np_it = table.as_numpy_iterator()
#     all_images_np = np.empty((0,(data_height * data_width + 1)))

#     for element in np_it:
#         # print("image shape:", element[0].shape)
#         flattened_image = np.reshape(element[0],(element[0].shape[0],-1))
#         flattened_image /= 255.0
#         reshaped_labels = np.expand_dims(element[1], 1)
#         image_label_np = np.append(flattened_image, reshaped_labels, axis=1)
#         all_images_np = np.append(all_images_np, image_label_np, axis=0)

#     # print("flattened shape:",all_images_np.shape)
#     column_names = list(range(all_images_np.shape[1]-1))
#     column_names.append('class')
#     df = pd.DataFrame(all_images_np, columns=column_names)
#     df['class'] = df['class'].astype(int)

#     return df


def get_image_data_as_df(ws, train_image_data_path, test_image_data_path, dataset_name):
    train_image_df = convert_to_table(train_image_data_path)
    test_image_df = convert_to_table(test_image_data_path)

    return train_image_df, test_image_df


# def get_data():

#     train_datagen = ImageDataGenerator(rescale=1./255)
#     train_gen = train_datagen.flow_from_directory(
#                 './Covid19-dataset/train',
#                 target_size=(data_width, data_height),
#                 batch_size=32,
#                 color_mode='grayscale')
    
#     test_datagen = ImageDataGenerator(rescale=1./255)
#     test_gen = test_datagen.flow_from_directory(
#                 './Covid19-dataset/test',
#                 target_size=(data_width, data_height),
#                 batch_size=4,
#                 color_mode='grayscale')

#     return train_gen, test_gen


def plot_data(image_generator):
    
    batch = next(image_generator)[0]
    
    num_images = batch.shape[0]
    cols = min(5,num_images)
    
    fig, ax = plt.subplots(nrows=1, ncols=cols)

    for i in range(cols):
        image = batch[i]
        ax[i].imshow(image)
        ax[i].axis('off')


# run = Run.get_context()

# height and width of the image to resize to
data_height = 100
data_width = int(data_height*1.3)


if __name__ == '__main__':
    get_data()

