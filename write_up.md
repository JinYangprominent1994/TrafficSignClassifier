# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributes

![alt text][image1]

I visualize an image randomly from training set and print its label in order to assert that the images and the labels are matching.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to shuffle the training set, which can be used to improve the performance of the CNN during the training process.

Then I convert the images to grayscale because the model can process gray images faster than processing color images.

Finally, I normalize the input data by using (pixel - 128)/128.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Here is an example of an original image and an augmented image:

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		  | 32x32x3 RGB image   							                          |
| Convolution 3x3     | 1x1 stride, valid padding, inputs 32x32x3, outputs 28x28x6 	|
| RELU					      |										                                          |
| Max pooling	      	| 2x2 stride, inputs 28x28x6, outputs 14x14x6                 |
| Convolution 3x3	    | 1x1 stride, valid padding, inputs 14x14x6, outputs 10x10x16 |
| RELU					      |									                                           	|
| Max pooling	      	| 2x2 stride, inputs 10x10x16, outputs 5x5x16                 |
| Flatten	           	| inputs 5x5x16, outputs 400                                  |
| Fully connected		  | inputs 400, outputs 120        									            |
| RELU					      |									                                           	|
| DROPOUT					    |	probilibity = 0.9 							                           	|
| Fully connected		  | inputs 120, outputs 84       									              |
| RELU					      |									                                           	|
| Fully connected		  | inputs 84, outputs 43        							                  |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I first define two placeholder for x and y, then I process y using one_hot.

After that, I use cross entropy and loss operation to reduce loss in order to optimize the model.

Next, I define a evaluate function to evaluate the validation accuracy, which is used during training process.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.947

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
-- The first architecture that I tried was LeNet, because it was

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
-- I added a dropout layer and adjusted hyper-parameters, such as sigma, learning rate and epoch.

* Which parameters were tuned? How were they adjusted and why?
-- I adjusted learning rate, sigma, epoch and keep_prob(which was is the probability that a node was closed in an epoch in the dropout layer). I did this, because, it could increase the validation accuracy.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
-- The dropout layer could help model avoid overfitting.

If a well known architecture was chosen:
* What architecture was chosen?
-- LeNet

* Why did you believe it would be relevant to the traffic sign application?
-- It is a popular architecture, and it has been tested by others.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
-- The validation accuracy is 0.947.
-- Furthermore, the accuracy of five images that I used to test is 80%. To be specific, the model recognize four images successfully from five images.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because the shape of the wrong image (label=1) is similar to that of the correct the image (label=5).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Vehicles over 3.5 metric tons prohibited   | Vehicles over 3.5 metric tons prohibited |
| Speed limit (30km/h)     			             | Speed limit (30km/h) 		            	  |
| Keep right				                         | Keep right								                |
| Turn right ahead	      	                 | Turn right ahead				              	  |
| Right-of-way at the next intersection		   | Right-of-way at the next intersection    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the validation set of 94.7%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is relatively sure that this is a Vehicles over 3.5 metric tons prohibited (probability of 0.49).

For the third image, the model is relatively sure that this is a Keep right (probability of 0.76).

The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.49                  | Vehicles over 3.5 metric tons prohibited      |
| 0.17     			        | Speed limit (30km/h) 		            	        |
| 0.76			            | Keep right								                    |
| 0.14	      	        | Turn right ahead				              	      |
| 0.26	                | Right-of-way at the next intersection         |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
