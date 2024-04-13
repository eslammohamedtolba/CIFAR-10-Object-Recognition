# CIFAR-10-Object-Recognition
This is a CIFAR-10 object classification model that uses a convolutional neural network (CNN) to classify images into ten different classes. 
![Image about the final project](<CIFAR-10 Object Recognition.png>)

## Prerequisites
Before running the code, make sure you have the following dependencies installed:
- NumPy
- OpenCV (cv2)
- Matplotlib
- Seaborn
- TensorFlow
- Keras

## Overview of the Code
1 -Load and preprocess the training data:
- Load the images and labels from tensorflow.keras.datasets.cifar10
- Split the data into input (X) and label (Y) data.
  
2 -Create a CNN model for image classification:
- Convolutional layers for feature extraction.
- Dense layers for classification.
- Softmax activation for the output layer.

3 -Compile and train the model using the training data.

4 -Evaluate the model's performance on the test data and display accuracy and loss.

5 -Visualize the model's accuracy and loss during training.

6 -Create a confusion matrix to evaluate the model's performance.

7 -Save the trained model for future use.

8-Create a predictive system that takes an image path as input, preprocesses the image, and uses the model to predict the class of the object in the image.


## Model Accuracy
The model achieves an accuracy of 72% on the test data with the CNN and 40% on the test data with the NN.

## Flask App Structure
- app.py: Contains Flask routes for rendering the web interface and handling predictions.
- templates/: Directory with HTML templates for the web pages.
- static/: Directory for static files as CSS.

## Contribution
Contributions to this project are welcome. You can help improve the model's accuracy, explore different CNN architectures, or enhance the data preprocessing and visualization steps. 
Feel free to make any contributions and submit pull requests.


