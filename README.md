# **Traffic Sign Recognition** 

## Introduction
This project aims to train the machine on Traffic signs and interpret the signs. This project helped me to understand the concepts of Deep learning , Tensor flow and CNN.
<br>

## Steps Followed

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
---

[//]: # (Image References)

[image1]: ./output/visualization_bar_chart.png "Visualization"
[image2]: ./output/before_greyscale.png "Before Grayscaling"
[image3]: ./output/after_greyscale.png "After Grayscaling"
[image4]:  ./test_images/0.jpg "Image0"
[image5]:  ./test_images/1.jpg "Image1"
[image6]:  ./test_images/2.jpg "Image2"
[image7]:  ./test_images/3.jpg "Image3"
[image8]:  ./test_images/4.jpg "Image4"
[image9]:  ./test_images/5.jpg "Image5"
[image10]: ./test_images/6.jpg "Image6"
[image11]: ./test_images/7.jpg "Image7"
[image12]: ./test_images/8.jpg "Image8"
[image13]: ./test_images/9.jpg "Image9"
[image14]: ./output/softmax_output.png "Prediction"

## Rubric Points

### Writeup / README

This document aims to serve as Writeup/README and here is a link to my [project code](https://github.com/rajeshrv2020/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set
<br>

| Parameter | Value |
| --- | --- |
| Size of training set | 34799 |
| Size of the validation set | 4410 |
| Size of test set | 12630 |
| Shape of a traffic sign image | (32,32) |
| number of unique classes/labels in the data set | 43 |

<br>

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1.Preprocessing the Image
The preprocessing of images are done as follows

* The image are converted into grey scale using the formula  Y = 0.299R + 0.587G + 0.114B
*


Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


![alt text][image3]



* The images are normalized for zero means and equal variance so that we get good gradient decent.


#### 2. Final model architecture 

My final model consisted of the following layers:
<br>


| Layer | Type | Description |
| --- | --- | --- |
| 0 | Input | 32x32x3 RGB image |
| 1 | Convolutional | 1x1 Stride, Input = 32x32x1. Output = 32x32x32 |
|   | RELU | |
|   | Max Pooling | 2x2 Stride, Input = 32x32x32. Output = 16x16x32 |
| 2 | Convolutional |  1x1 Stride, Input = 32x32x32. Output = 32x32x32 |
|   | RELU | |
|   | Max Pooling | 2x2 Stride, Input = 32x32x32. Output = 16x16x32 |
| 3 | Convolutional | 1x1 Stride, Input = 32x32x32, Output = 8x8x32 |
|   | RELU | |
|   | Dropout| |
| 4 | Convolutional | 1x1 Stride, Input = 8x8x32, Output = 8x8x32 |
|   | RELU | |
|   | Dropout| |
| 5 | Flatten | Input = 8x8x32. Output = 2048 |
| 6 | Fully Connected | Input = 2048. Output = 128. |
|   | RELU | |
|   | Dropout| |
| 7 | Fully Connected | Input = 128. Output = 84. |
|   | RELU | |
|   | Dropout| |
| 8 | Fully Connected | Input = 84. Output = 43. |
|   | Classifier | |

<br>

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following

| Name | Type |
| --- | --- |
| optimizer | AdamOptimizer |
| batch size | 128 |
| number of epochs | 20 |
| learning rate | 0.001 |


#### 4.Approach taken for finding a solution

My final model results were:

| Dataset Accuracy | Value |
| --- | --- |
| Training accuracy | 0.982 |
| Validation accuracy | 0.983 |
| Test accuracy | 0.953 |

If a well known architecture was chosen:

* LeNet Architecture was chosed without much iteration.


### Test a Model on New Images

#### 1. Images chosen for Test

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13] 

The Image has high variable parameters which might be little difficult to predict
* Image 2,4,7 is very dark
* Image 10 has a shadow 
* Image 1 and 5 background has lot of variations
* Image 7 has high contrast.

<br>

#### 2. Models Prediction for the new Test images

Here are the results of the prediction:
![alt text][image14]

The model was able to correctly predict 10 of the 10 traffic signs, which gives an accuracy of 100%.


