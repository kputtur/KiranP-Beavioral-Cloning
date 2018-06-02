import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

from PIL import Image
import math
from random import shuffle

path2data = '/home/kputur/Desktop/simulator/train_1'


def load_images_and_measurements:
car_images = []

steering_angles  = []
	 with open(path2data + 'driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			steering_center = float(row[3])
	
			#create adjusted steering measurements for the side camera images
			correction = 0.2
			steering_left = steering_center + correction
			steering_right = steering_center - correction


			img_center = np.asarray(Image.open(path2data + row[0])))
			img_left = np.asarray(Image.open(path2data + row[1])))
			img_right = np.asarray(Image.open(path2data + row[2])))
	
		 	car_images.extend(img_center, img_left, img_right)
			steering_angles.extend(steering_center, steering_left, steering_right)
	 return car_images, steering_angles	


def flip_image_steering(image, steering_angle):
	flipped_image = np.fliplr(image)
	flipped_steering_angle = steering_angle * -1.0
	return flipped_image, flipped_steering_angle


def augment_data (images, measurements):
	augmented_images, augmented_measurements = [], []

	for image, steering_angle in zip(images, measurements):
		flipped_image, flipped_steering_angle = flip_image_steering(image, steering_angle)
		augmented_images.append(flipped_image)
		augmented_images.append(image)
		augumented_measurements.append(flipped_steering_angle)
		augmented_measurements.append(steering_angle)
		
	return augmented_images, augmented_measurements



X_train = np.array(augumented_images)
y_train = np.array(augumented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D

from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
	


