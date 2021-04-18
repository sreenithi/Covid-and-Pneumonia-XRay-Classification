*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

*TODO:* Write a short introduction to your project.
The aim of this project is to classify Chest X-Ray images as whether they are positive for Covid-19, Viral Pneumonia or Normal (free of both diseases). Two types of classification was implemented using Microsoft Azure ML as follows:
 1. The first is to train a Convolutional Neural Network (CNN) on the dataset and use the HyperDrive feature of AzureML to find the optimal number of filters for each Conv2D layer of the model.
 2. The second option was to use the AutoML feature of AzureML with DNN enabled to automatically train multiple classifier models on the data

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.
The data used for this project is the [Covid-19 Image Dataset](https://www.kaggle.com/pranavraikokte/covid19-image-dataset) from Kaggle. It is an image dataset with 3 classes for *Covid*, *Viral Pneumonia* and *Normal* chest x-rays. 

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
The task we are trying to solve is to classify the Chest X-Rays in the dataset as whether they are affected by *Covid*, *Viral Pneumonia* or not affected by either (*Normal*).

#### Featurization of the data
Some common image processing techniques were used on the images as follows:
 * Resizing to a smaller size image (256x256 for HyperDrive) and (100x130 for AutoML)
 * Converting to color mode *grayscale*
 * Scaling the values to be in the range of 0 and 1 (where 0 corresponds to black, and 1 corresponds to white)


### Access
*TODO*: Explain how you are accessing the data in your workspace.
The dataset downloaded from Kaggle is first uploaded to the project directory. Once uploaded, the directory structure of the data folder inside the project directoryis as follows:
 - Covid-and-Pneumonia-XRay-Classification (Main Project Folder)
   - Covid19-dataset
     - train
       - Covid
         ... (image files)
       - Viral Pneumonia
         ... (image files)
       - Normal
         ... (image files)
     - test
       - Covid
         ... (image files)
       - Viral Pneumonia
         ... (image files)
       - Normal
         ... (image files)
   - ... (other .py and .ipynb files)

The data was read from this folder and accessed in two different forms for the Hyperdrive and AutoML:
 1. For the HyperDrive, the image data from the *Covid19-dataset/train* folder was read, processed as discussed in the previous section and used to train the CNN. In addition, the data was also registered as a FileDataset.
 2. For the AutoML, as AzureML's AutoML feature currently does not support FileDatasets, the image data from the *Covid19-dataset/train* folder was read, processed as discussed in the previous section and further converted to a Pandas Dataframe, which was then registered as a TabularDataset. This registered dataset was used to run the AutoML experiment.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
The AutoML settings used for this project is as follows:
*{"enable_dnn":True, "experiment_timeout_minutes":45, "max_cores_per_iteration":3, "enable_tf":False, "featurization":"off"}*
As this project uses an image dataset with a Convolutional Neural Network chosen for the HyperDrive, similar Deep Learning algorithms had to be considered for the AutoML as well. Hence settings such as *"enable_dnn":True, "enable_tf":False* were used to enable Deep Learning models in the AutoML.
But at the same time, to restrict the long training time of AutoML, the *experiment_timeout_minutes* was set to 45, and *featurization* was disabled.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*********** Run best_run.get_output() to fill this section ************

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
