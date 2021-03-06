# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia-architecture]: ./examples/nvidia-architecture.png "Nvidia network Architecture"
[center-ride]: ./examples/center_ride.jpg "Center ride"
[left_ride]: ./examples/left_ride.jpg "Left ride"
[left_to_center]: ./examples/left_to_center.jpg "Left to center"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* Behavioral-Cloning.ipynb containing the same model.py code
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5  filter sizes and depths between 24 and 64 (model.py lines 78-93) 

Convolution layers have RELU activations to introduce nonlinearity (code line 82-86), and the data is normalized in the model using a Keras lambda layer (code line 79). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers with probability 0.7 in order to reduce overfitting (model.py lines 90, 92). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road also I recored slow turns on heavy turns with different positions of car on track (center, left and right side). In addition, I recorded a few laps from track 2. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia network [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) I thought this model might be appropriate because it was already tested on real world usecases and shows good performance.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model with a few Dropout layes and added data from track 2.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (sharp turns). To improve the driving behavior in these cases, I recorded driving in these places at low speed
and differet position of a car on track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road with speed 30 (max).

#### 2. Final Model Architecture

The final model architecture (model.py lines 78-98) consisted of a convolution neural network with the following layers and layer sizes:

![alt text][nvidia-architecture]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps (two in one direction and two in another one)  on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center-ride]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to return to center position after some turns. These images show what a recovery looks like starting from left side to center :

![alt text][left_ride]
![alt text][left_to_center]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help to get more  generalized model. Also I used images form left and right cameras with adjusted steering angle for each side.



After the collection process, I had 37886 number of data points. I then preprocessed this data by cropping images 50 pixels from the top and 20  pixels from the bottom of the image and resizing images to 66x200 (original nvidia network input size)


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
I took very high number of epochs 40, but it shows the best result even with such small dataset. I used an adam optimizer so that manually training the learning rate wasn't necessary.
