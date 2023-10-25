# import kaggle module
!pip install kaggle
# configuring the path of Kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
# API to fetch the dataset from Kaggle
kaggle competitions download -c cifar-10
# unzip the zip folder of images
! unzip '/content/face-mask-dataset.zip'
# install py7zr to unzip the 7z train file
!pip install py7zr


# import required dependencies
import py7zr
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dropout,Dense
from tensorflow.math import confusion_matrix
import seaborn as sns
import pickle as pk


# unzip the train data file
archive = py7zr.SevenZipFile('cifar-10/train.7z','r')
archive.extractall()
archive.close()


# Load the labels of training data
label_data = pd.read_csv('cifar-10/trainLabels.csv')
# Show the first five rows in the label data and print its size
label_data.head()
len(label_data)


# Encode the labels of training data
encoder = LabelEncoder()
label_data['label'] = encoder.fit_transform(label_data['label'])
# Create dictionary for the label and its encoded label that created by label encoder
classes = {encoded_label:label for encoded_label,label in zip(np.unique(label_data['label']),encoder.classes_)}
print(classes)


# Show the first and last five samples in the data
label_data.head()
label_data.tail()


# extract the images names from the train folder
images_names = os.listdir('train')
# Show the first 10 images to check if the images have the same size or not
i = 0
for image_name in images_names:
    image_path = os.path.join('train',image_name)
    image = cv2.imread(image_path)
    print(image.shape)
    if i==10:
        break
    i+=1

# Load the images and its labels in the data list
data = []
for image_name,label in zip(images_names,label_data['label']):
    image_path = os.path.join('/kaggle/working/train',image_name)
    image = cv2.imread(image_path)
    if image is not None:
        data.append([image,label])

# Shuffle the data list
random.shuffle(data)


# split the data into input and label data
X = []
Y = []
for feature,label in data:
    X.append(feature)
    Y.append(label)

# Convert input and label data into numpy array
X = np.array(X)
Y = np.array(Y)

# Show the first image before scaling the images
print(X[0])
# Scaling input data
X = X/255
# Show the first image after scaling the images
print(X[0])

# Show the shape of input and label data
print(X.shape)
print(Y.shape)


# Split input and label data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.9,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)


# Create the model
input_size = (32,32,3) # Determine input size
# Determine number of classes
num_classes = 10

# Create the model
Model = Sequential([
    Input(shape=input_size),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
# compile the model 
Model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# train the model with train input and label data
result = Model.fit(x_train,y_train,epochs=30,validation_split=0.1)

# Visualize the accuracy with the validation accuracy
plt.figure(figsize=(7,7))
plt.plot(result.history['accuracy'],color='red')
plt.plot(result.history['val_accuracy'],color='blue')
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['accuracy','val_accuracy'],loc='lower right')
# Visualize the loss with the validation loss
plt.figure(figsize=(7,7))
plt.plot(result.history['loss'],color='red')
plt.plot(result.history['val_loss'],color='blue')
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['loss','val_loss'],loc='upper right')

# evaluate the model
evaluation = Model.evaluate(x_test,y_test)
print("the loss value is: ",evaluation[0])
print("the accuracy value is: ",evaluation[1])

# Make the model predict on test input data
predicted_y = Model.predict(x_test)
y_predicted_values = []
for value in predicted_y:
    y_predicted_values.append(np.argmax(value))
comparison = []
for predicted_value,true_value in zip(y_predicted_values,y_test):
    comparison.append([predicted_value,true_value])
print(comparison)
# create the confusion matrix and visualize it
plt.figure(figsize=(5,5))
conf_matrix = confusion_matrix(y_test,y_predicted_values)
sns.heatmap(conf_matrix,square=True,cbar=True,annot=True,annot_kws={'size':8},cmap='Blues')


# Save the model
pk.dump(Model,open('trained_model.sav','wb'))


# Make a predictive system 
image_path = input("Enter the image path: ")
image = cv2.imread(image_path)
# Show the image
plt.imshow(image)
# Ensure the image has 3 color channels (e.g., convert from grayscale to RGB)
if image.shape[-1] == 1:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# Normalize the image by scaling it
image = image / 255
# Make the model predict what is in the image if it's dog will print 1 otherwise will print 0
prediction = Model.predict(np.expand_dims(image, axis=0))
predicted_class = np.argmax(prediction)
# Replace the encoded label with the original label 
print(classes[predicted_class])


