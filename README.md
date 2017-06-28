# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/bar.png "Visualization"
[image4]: ./traffic-signs-news/00000_00014.jpg "Traffic Sign 1"
[image5]: ./traffic-signs-news/00000_00023.jpg "Traffic Sign 2"
[image6]: ./traffic-signs-news/00001_00017.jpg "Traffic Sign 3"
[image7]: ./traffic-signs-news/00004_00019.jpg "Traffic Sign 4"
[image8]: ./traffic-signs-news/2.png "Traffic Sign 5"
[image9]: ./traffic-signs-news/00005_00000.jpg "Traffic Sign 6"
[image10]: ./traffic-signs-news/00005_00003.jpg "Traffic Sign 7"
[image11]: ./traffic-signs-news/00005_00029.jpg "Traffic Sign 8"
[image12]: ./traffic-signs-news/00006_00026.jpg "Traffic Sign 9"
[image13]: ./traffic-signs-news/00006_00028.jpg "Traffic Sign 10"
[image14]: ./traffic-signs-news/00008_00012.jpg "Traffic Sign 11"
[image15]: ./traffic-signs-news/00009_00029.jpg "Traffic Sign 12"
[image16]: ./traffic-signs-news/00012_00018.jpg "Traffic Sign 13"
[image17]: ./traffic-signs-news/00020_00024.jpg "Traffic Sign 14"
[image18]: ./traffic-signs-news/00023_00026.jpg "Traffic Sign 15"
[image19]: ./traffic-signs-news/00025_00025.jpg "Traffic Sign 16"
[image20]: ./report/norma.png "Normalized data"
[image21]: ./report/data.png "Raw data"
[image22]: ./traffic-signs-news/00025_00025.jpg "To"
[image22]: ./traffic-signs-news/00025_00025.jpg "To"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (34799, 32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images over the labels.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocess part

As a last step, I normalized the image data using openCV min max normalizer to get a better contrast.

Here is an example a sample of normalized images:

![alt text][image20]


#### 2. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Normalization  		|         									    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Normalization  		|         									    |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten       		| output 400  									|
| Fully connected		| output 120   									|
| Fully connected		| output 84  									|
| Dropout  				| probability : 0.5								|
| Softmax				| outpur 43    									|
|						|												|

#### 3. How I trained my model.

The model is based on Lenet with some inspiration from Cifar-10 such as include normalization layers. The learning rate is 0.0005 to avoid being stuck in local optimum. The number of epochs is 15 to compensate a small learning rate. I used the Adam optimizer with a batch size of 128 to avoid memory issues. 

#### 4. My approach 

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.964 
* test set accuracy of 0.947

I used an iterative approach, I started my project with a Lenet architecture, then add features from Cifar-10 (normalization layers at strategic points, I tried after the first pooling layer and after the second conv layer) and then add a last dropout layer. The first achitecture (Lenet) had an accuracy of 0.85, the next one had 0.93 (with normalizers) and finaly 0.96 with dropout due to possible overfitting. I also tried a valid padding to get smaller outputs to accelerate the trainning without too much loss. I think the first convolution layer with valid padding is a good thing to the neural network to focus on the center of the image. Combine convolution layers with pooling layers is also very effective.

### Test a Model on New Images

Here are 16 German traffic signs that I found on the web, from http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I think the image with 30km/h speed limit should be difficult to classify because it's very noisy.

#### 2. model's predictions on these new traffic signs

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      			| Yield		   									| 
| Yield      			| Yield		   									| 
| Bumpy road   			| Bumpy road									|
| Slippery road			| Slippery road									|
| Speed limit (100km/h)	| Speed limit (80km/h)			 				|
| Roundabout mandatory	| Roundabout mandatory 							|
| General caution		| General caution 								|
| Right-of-way at the next intersection	| Right-of-way at the next intersection|
| Speed limit (70km/h)	| Speed limit (70km/h) 							|
| Vehicles over 3.5 metric tons prohibited	| End of no passing 		|
| No entry				| No entry 										|
| Priority road	        | Priority road 								|
| Ahead only			| Ahead only 									|
| Speed limit (30km/h)	| Speed limit (30km/h) 							|
| Traffic signals		| Traffic signals 								|



The model was able to correctly guess 14 of the 16 traffic signs, which gives an accuracy of 87.5%. This is less than previous accuracies from test, validation and train sets.

#### 3. How certain the model is when predicting on each of the 16 new images by looking at the softmax probabilities for each prediction. 

The code for making predictions on my final model is located in the 17th cell of the Jupyter notebook.

Finaly, the sign 30km/h speed limit was not a problem compared to 100km/h speed limit sign. The top five soft max probabilities were:

Yield was predicted :
Yield at 13.816567420959473%
Speed limit (60km/h) at 2.8390212059020996%
Stop at 1.4006872177124023%

Yield was predicted :
Yield at 17.291114807128906%
Priority road at 2.974719524383545%
No passing for vehicles over 3.5 metric tons at -0.9347158670425415%

Stop was predicted :
Stop at 10.012700080871582%
No vehicles at 2.011249542236328%
No entry at 1.8586984872817993%

Bumpy road was predicted :
Bumpy road at 7.810288906097412%
Road work at 4.179072380065918%
Bicycles crossing at 2.8104777336120605%

Slippery road was predicted :
Slippery road at 9.70471477508545%
Children crossing at 3.6869354248046875%
Beware of ice/snow at 1.8858410120010376%

Speed limit (100km/h) was predicted :
Speed limit (80km/h) at 7.455338001251221%
Speed limit (100km/h) at 5.580087661743164%
Speed limit (120km/h) at 1.863337516784668%

Roundabout mandatory was predicted :
Roundabout mandatory at 5.858547210693359%
Go straight or left at 1.4140509366989136%
Keep left at -0.16061408817768097%

General caution was predicted :
General caution at 14.54281234741211%
Pedestrians at 7.883150100708008%
Traffic signals at 7.646287441253662%

Right-of-way at the next intersection was predicted :
Right-of-way at the next intersection at 12.881926536560059%
Slippery road at 6.561455726623535%
Beware of ice/snow at 4.8460869789123535%

Speed limit (70km/h) was predicted :
Speed limit (70km/h) at 10.193379402160645%
Speed limit (30km/h) at 5.222599506378174%
Speed limit (80km/h) at 4.329324722290039%

Vehicles over 3.5 metric tons prohibited was predicted :
End of no passing at 6.396401405334473%
Dangerous curve to the right at 3.9751317501068115%
Vehicles over 3.5 metric tons prohibited at 3.8834900856018066%

No entry was predicted :
No entry at 16.12698745727539%
Stop at 6.474311351776123%
Priority road at -0.2244900017976761%

Priority road was predicted :
Priority road at 8.426002502441406%
Yield at 3.1253669261932373%
No entry at 0.7640930414199829%

Ahead only was predicted :
Ahead only at 10.61231803894043%
Turn left ahead at -0.07316013425588608%
Speed limit (80km/h) at -0.6469821929931641%

Speed limit (30km/h) was predicted :
Speed limit (30km/h) at 10.765523910522461%
Speed limit (50km/h) at 7.358022212982178%
Speed limit (70km/h) at 5.211430549621582%

Traffic signals was predicted :
Traffic signals at 6.037755012512207%
General caution at 4.119287014007568%
Pedestrians at 2.744610071182251%

The sign "Vehicles over 3.5 metric tons prohibited" wasn't predicted correctly, I think it's due to the short number of this sign labelised.
