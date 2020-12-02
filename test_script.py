import pandas as pd
from tensorflow import keras
from keras import utils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
import numpy as np
import os
import cv2
import tensorflow as tf

def prepare_image(img):
    #convert the color from
    #BGR to RGB then convert to PIL #array
    cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
 
    # resize the array (image) into 224*224 and then PIL image
    im_resized = im_pil.resize((224, 224)) 
    img_array = image.img_to_array(im_resized)
    return img_array

def load_images(folder,x,y,df):
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,file))
        if(img is None):
            continue
        img = prepare_image(img)
        img = preprocess_input(img)
        img = img.reshape(1,-1)
        
        label=df[df["Image_name"]==file].Label
        
        if(x.shape[0]==0):
            x=np.hstack(img)
            y=np.hstack(label)
        else:
            x=np.vstack((x,img))
            y=np.vstack((y,label))
    return x,y

##Trimmimg the image names in .xlsx file
def str_trim(x):
    return str(x).strip()

##Loading the excel files
digitalBlur_df=pd.read_excel("D:/CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx")
naturalBlur_df=pd.read_excel("D:/CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx")

##Naming the columns of Dataframe as same
digitalBlur_df.columns=["Image_name","Label"]
naturalBlur_df.columns=["Image_name","Label"]

##Making sure Image_name column is correct
naturalBlur_df["Image_name"]=naturalBlur_df["Image_name"]+".jpg"

##Trimming image names
naturalBlur_df["Image_name"]=naturalBlur_df["Image_name"].apply(str_trim)
digitalBlur_df["Image_name"]=digitalBlur_df["Image_name"].apply(str_trim)

##Storing the correct labels 
digitalBlur_df["Label"]=digitalBlur_df["Label"].apply(lambda x: x if x==1 else 0)
naturalBlur_df["Label"]=naturalBlur_df["Label"].apply(lambda x: x if x==1 else 0)

x_test=np.array([])
y_test=np.array([])

##Loading Evaluation_set images
x_test,y_test=load_images("D:/CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet",x_test,y_test,naturalBlur_df)
print("\nEvaluationSet/NaturalBlurSet images loaded successfully\n")

x_test,y_test=load_images("D:/CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet",x_test,y_test,digitalBlur_df)
print("\nEvaluationSet/DigitalBlurSet images loaded successfully\n")

##Reshaping test_Set according to the model
x_test = x_test.reshape(x_test.shape[0],224,224,3)
y_test = utils.to_categorical(y_test)

##Loading the saved model
model = keras.models.load_model("model")

##Testing the data over loaded model
test_loss,test_acc = model.evaluate(x_test,y_test)
print("test_loss : ",test_loss,"\ntest_accuracy : ",test_acc)
