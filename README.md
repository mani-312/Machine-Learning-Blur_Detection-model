# Machine-Learning-Blur_Detection-model
Detecting an image is a blur or not has become a challenging and important aspect to the humans.This has several use cases in the industrial market.One of those can be like a software automatically detecting the uploaded image as blur or clear.

## Dataset details:
You can download the **CERTH_ImageBlurDataset** at http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip.

Contents in the dataset:
  Training set

       630 undistorted images 

       220 naturally-blurred images

       150 artificially-blurred images

  Evaluation set consisting of the “natural blur” set and of the “artificial blur” set.

       Natural blur set

          589 undistorted images

          411 naturally-blurred images

       Artificial blur set

          30 undistorted images

          450 artificially-blurred images
          
# Convolutional Neural Network Model creation
Model includes Conv2D(), MaxPooling(), Flatten(), Dropout() and Dense() layers.
# Model Accuracy
Got the accuracy of **80%** on evaluation set.
