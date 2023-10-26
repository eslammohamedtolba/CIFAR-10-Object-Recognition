# CIFAR-10-Object-Recognition
This is a CIFAR-10 object classification model that uses a convolutional neural network (CNN) to classify images into ten different classes. 

## Prerequisites
Before running the code, make sure you have the following dependencies installed:
- Kaggle API
- py7zr
- NumPy
- OpenCV (cv2)
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- TensorFlow
- Keras

## Overview of the Code
1-Download and preprocess the CIFAR-10 dataset:
- Download the dataset from Kaggle using the Kaggle API.
- Extract the dataset files using py7zr.
- Load the labels of the training data and encode them.

2-Load and preprocess the training data:
- Load the images and labels.
- Shuffle the data.
- Split the data into input (X) and label (Y) data.
- Normalize the pixel values of the images.

3-Split the input and label data into training and test data.

4-Create a NN model for image classification:
- Dense layers for classification.
- sigmoid activation for the output layer.
  
5-Create a CNN model for image classification:
- Convolutional layers for feature extraction.
- Dense layers for classification.
- Softmax activation for the output layer.

6-Compile and train the model using the training data.

7-Evaluate the model's performance on the test data and display accuracy and loss.

8-Visualize the model's accuracy and loss during training.

9-Create a confusion matrix to evaluate the model's performance.

10-Save the trained model for future use.

11-Create a predictive system that takes an image path as input, preprocesses the image, and uses the model to predict the class of the object in the image.


## Model Accuracy
The model achieves an accuracy of 70% on the test data with the CNN and 40% on the test data with the NN.

## Contribution
Contributions to this project are welcome. You can help improve the model's accuracy, explore different CNN architectures, or enhance the data preprocessing and visualization steps. 
Feel free to make any contributions and submit pull requests.


