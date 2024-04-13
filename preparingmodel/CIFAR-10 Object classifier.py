# Load dependencies
# import required dependencies
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2
from tensorflow.keras.models import save_model
from tensorflow.keras import datasets


# Load the training and testing data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# Show shape of training and testing data
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# Get the classes names
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
# Determine input size for the NN and CNN
input_size = (32,32,3) 
# Determine number of classes
Num_classes = 10


def PrintFirstNum(num):
    index = num
    for feature, label in zip(x_train, y_train):
        print(classes[label[0]])
        plt.imshow(feature)
        plt.show()
        index += 1
        if index == 5:
            break

# Show the first five images
PrintFirstNum(5)

# Create the CNN model
CNNModel = Sequential([
    Input(shape=input_size),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(64,activation='relu'),
    Dense(Num_classes,activation='softmax')
])
# Compile the CNNModel
CNNModel.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# Train the model with train input and label data
CNNresult = CNNModel.fit(x_train,y_train,epochs=10,validation_split=0.1)
# Evaluate the model
evaluation = CNNModel.evaluate(x_test,y_test)
print("the loss value is: ",evaluation[0])
print("the accuracy value is: ",evaluation[1])


# Visualize the accuracy with the validation accuracy
plt.figure(figsize=(7,7))
plt.plot(CNNresult.history['accuracy'],color='red')
plt.plot(CNNresult.history['val_accuracy'],color='blue')
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['accuracy','val_accuracy'],loc='lower right')
# Visualize the loss with the validation loss
plt.figure(figsize=(7,7))
plt.plot(CNNresult.history['loss'],color='red')
plt.plot(CNNresult.history['val_loss'],color='blue')
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['loss','val_loss'],loc='upper right')


# Make the model predict on test input data
predicted_y = CNNModel.predict(x_test)
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
# Create a classification report
class_report = classification_report(y_test,y_predicted_values)
print(class_report)


# Save the model
save_model(CNNModel, 'CNNModel.h5')

# Make a predictive system 
def predictive_system(path):
    image_path = path
    image = cv2.imread(image_path)
    # Show the image
    plt.imshow(image)
    # Ensure the image has 3 color channels (e.g., convert from grayscale to RGB)
    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Make the model predict what is in the image if it's dog will print 1 otherwise will print 0
    prediction = CNNModel.predict(np.expand_dims(image, axis=0))
    predicted_class = np.argmax(prediction)
    # Replace the encoded label with the original label 
    print(classes[predicted_class])




