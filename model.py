import pandas as pd
from tensorflow import keras
from keras import utils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
import numpy as np
import os
import cv2
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get("accuracy")>0.95):
            print("\nReached 95% accuracy\n")
            self.model.stop_training=True

def prepare_image(img):
    #convert the color from
    #BGR to RGB then convert to PIL #array
    cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
 
    # resize the array (image) into 224*224 and then PIL image
    im_resized = im_pil.resize((224, 224)) 
    img_array = image.img_to_array(im_resized)
    return img_array


def load_images(folder,data):
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,file))
        if(img is None):
            continue
        #preprocessing the image
        img = prepare_image(img)
        img = preprocess_input(img)
        img = img.reshape(1,-1)
        
        if(data.shape[0]==0):
            data = np.hstack(img)
        else:
            data = np.vstack((data,img))
    return data

###Loading the training set
data = np.array([])

x_non_blur = load_images("CERTH_ImageBlurDataset/TrainingSet/Undistorted",data)
y_non_blur = np.zeros((x_non_blur.shape[0],1))
print("\nTrainingSet/Undistorted images loaded successfully\n")

x_blur = load_images("CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred",data)
print("\nTrainingSet/Artificially-Blurred images loaded successfully\n")

x_blur = load_images("CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred",x_blur)
print("\nTrainingSet/Naturally-Blurred images loaded successfully\n")

x_blur = load_images("CERTH_ImageBlurDataset/TrainingSet/NewDigitalblur",x_blur)
print("\nTrainingSet/NewDigitalBlur images loaded successfully\n")

y_blur = np.ones((x_blur.shape[0],1))

##Vertically stacking blur and non_blur data of train_set
x = np.vstack((x_non_blur,x_blur))
y = np.vstack((y_non_blur,y_blur))

#Shuffling the train data
x , y = suffle(x,y)

###Model structure using CNN
model=tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(224,224,3))
	])
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(128,(3,3),padding="same",activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(256,activation="relu"))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(2,activation="softmax"))

model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

callbacks=myCallback()

##Reshaping train_Set according to the model
x = x.reshape(x.shape[0],224,224,3)
y = utils.to_categorical(y)

##Training the data using callbacks
history = model.fit(x,y,epochs=10,callbacks=[callbacks])

print("\nModel_Loss == "+history.history["loss"][-1])
print("\nModel_Accuracy == "+history.history["accuracy"][-1])

accuracy = history.history["accuracy"]
loss = history.history["loss"]
epochs = len(history.history["accuracy"])

plt.subbplot(1,2,1)
plt.plot(range(epochs),accuracy)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(range(epochs))
plt.yticks([])

plt.subbplot(1,2,2)
plt.plot(range(epochs),loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(epochs))
plt.yticks([])
