import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data_path = 'record'


samples = []
with open( data_path + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                def get_image_path(row):
                    return data_path + '/IMG/'+batch_sample[row].split('/')[-1]
                
                def read_image(path):
                    img = cv2.imread(path)
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                center_image_path = get_image_path(0) 
                left_image_path =  get_image_path(1)
                right_image_path =  get_image_path(2)
                
                center_image = read_image(center_image_path)
                left_image = read_image(left_image_path)
                right_image = read_image(right_image_path)
                
                correction = 0.25 # this is a parameter to tune
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                
                fliped_center_image = cv2.flip(center_image, 1)
                fliped_center_angle = center_angle*-1.0
                
                images.extend((center_image, left_image, right_image, fliped_center_image))
                angles.extend((center_angle, left_angle, right_angle, fliped_center_angle))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Convolution2D, Flatten, Dense, Dropout
import tensorflow as tf
import cv2

def resize_image(x):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(x, (66, 200))

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Lambda(resize_image))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation ="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation ="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation ="relu"))
model.add(Convolution2D(64,3,3,  activation ="relu"))
model.add(Convolution2D(64,3,3,  activation ="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.7))
model.add(Dense(10))
model.add(Dropout(0.7))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.summary()
history_object = model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=40)


model.save("model.h5")  