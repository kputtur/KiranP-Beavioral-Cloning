import csv
import os
import cv2
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.model_selection import train_test_split

from PIL import Image
import gc
from random import shuffle

from keras.models import Sequential  
from keras.layers import Convolution2D, Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers import MaxPooling2D

import json
from keras.models import model_from_json, load_model


def load_csv_file(path):
  lines = [] 
  file_name = path + '/driving_log.csv'
  try:
      with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lines.append(row)
  except IOError:
      print("Some Error Occured")
      raise
  return lines

def remove_zeros(lines):
    i = 0
    while i < len(lines):
        if (abs(float(lines[i][3])) < PARAM['steer_angle_tolerance']):
            #removing random values
            if np.random.rand() < 0.8:
                del lines[i]
        i += 1
    return lines


def plot_loss(model_history):
        print(model_history.history.keys())
        plt.plot(model_history.history['loss'])
        plt.plot(model_history.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plot_img_name = PARAM['save_model_file']
        plt.savefig(plot_img_name + '.png')
        plt.show()
        

def load_images_and_measurements(lines):
    car_images = []
    steering_angles  = []
    
    for row in lines:
        steering_center = float(row[3])

        #create adjusted steering measurements for the side camera images
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        
        center_path = PARAM['path2data'] + '/IMG/' + row[0].split('/')[-1]
        left_path = PARAM['path2data'] + '/IMG/' + row[1].split('/')[-1]
        right_path = PARAM['path2data'] + '/IMG/' + row[2].split('/')[-1]
        
        img_center = np.asarray(Image.open(center_path))
        img_left = np.asarray(Image.open(left_path))
        img_right = np.asarray(Image.open(right_path))

        car_images.extend([img_center, img_left, img_right])
        steering_angles.extend([steering_center, steering_left, steering_right])

    return car_images, steering_angles    


def flip_image_steering(image, steering_angle):
    flipped_image = np.fliplr(image)
    flipped_steering_angle = steering_angle * -1.0
    return flipped_image, flipped_steering_angle


def augment_data (images, measurements):
    augmented_images, augmented_measurements = [], []

    for image, steering_angle in zip(images, measurements):
        flipped_image, flipped_steering_angle = flip_image_steering(image, steering_angle)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
	#update flipped image and measurements
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_steering_angle)

	#update	HSV image and measurement
        augmented_images.append(image)
        augmented_measurements.append(steering_angle)
        
    return augmented_images, augmented_measurements


def data_generator(samples, batch_size):
    print('data generator seeding ...')
    num_samples = len(samples)
    
    while 1:        #Starting the co-routine
        shuffle(samples)
        
        X_data = []
        y_data = []

        for i, line in enumerate(samples):
            images, measurements = load_images_and_measurements([line])
            
            #flip the image file
            augmented_images, augmented_measurements = augment_data(images, measurements)

            #Add the generated data into yeild array
            X_data.extend(augmented_images)
            y_data.extend(augmented_measurements)


             #Check if i is equal to 
            if i == (num_samples - 1) or  len (X_data) > batch_size:
                yield sklearn.utils.shuffle(np.array(X_data[:batch_size]), np.array(y_data[:batch_size]))
                X_data[batch_size:]
                y_data[batch_size:]    

             


def nvidia_model(summary=True):
    model = Sequential()
    model.add(Lambda(lambda x: x/255. - 0.5,  input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20),(0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2),   activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    if summary:
        model.summary()


    return model

def LeNet_model(summary=True):
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(PARAM['input_width'], PARAM['input_height'], PARAM['input_channels'])))
    model.add(Cropping2D(cropping=(PARAM['cropping'])))
    #MaxPooling 2D Layer
    model.add(MaxPooling2D())
        #Convlution Layer
    model.add(Convolution2D(6, (3, 3), activation='relu'))
    #Maxpooling 2D layer
    model.add(MaxPooling2D())
    
    #Flatten
    model.add(Flatten())
    #First Dense Layer
    model.add(Dense(120))
        # Second Dense Layer
    model.add(Dense(84))    
    
    # Single Dense layer
    model.add(Dense(1))

    
    if summary:
           model.summary()

    return model


def load_visualize_model(model=None):
           from keras.utils import plot_model
           plot_model(model, to_file='model_loaded.png')
           


if __name__ == '__main__':
  
    PARAM = {
                'steer_angle_tolerance':0.04,
                 'batchsize': 256,
                 'input_width': 160,
                 'input_height': 320,
                 'input_channels': 3,
                 'correction': 0.15,
                 'cropping': ((50,20),(0,0)),
                 'epoch' : 10,
                 'save_model_file' : 'nvidia-newer',
                 'chosen_model' : 'nvidia',
                 'path2data': '/home/kputtur/Desktop/simulator',
                 'samples_per_epoch': 512,
                 'recursive': 0
              }
   
    parser = argparse.ArgumentParser(description='Behavioral Cloning Project')
    parser.add_argument('-l', '--load_model', type=str, required=False, default=None,
                        help='Path to model definition for loading h5 model')
    parser.add_argument('-s', '--save_model', type=str, required=False, default=None,
                        help='Path to model definition for saving (without json/h5 extension)')
    parser.add_argument('-m', '--choose_model', type=str, required=False, default='nvidia',
                        help='choose either nvidia or lenet in future custom(WIP) model')
    parser.add_argument('-t', '--training_data', type=str, required=False, default=None,
                        help='Path to folder with driving_log.csv and IMG subfolder')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=10,
                        help='Number of Epochs to get trained')
    parser.add_argument('-b', '--batchsize', type=int, required=False, default=128,
                        help='Number of batches')
    parser.add_argument('-te', '--train_epochs', type=int, required=False, default=1280,
                        help='Number of train_steps_per_epoch')
    parser.add_argument('-r', '--recursive', type=int, required=False, default=0,
                        help="Specify recursive directory where IMG file needs to be ")
    args = parser.parse_args()

    model = None
    if args.load_model:
      print("Loading model from  " + args.load_model)
      #model = load_model(args.load_model)
      load_visualize_model(model=args.load_model)
      
    model_file = None
    if args.save_model:
      print("setting save model as  " + args.save_model)
      PARAM['save_model_file'] = args.save_model
      
    
    if args.choose_model:
      print("Chosen model to Train  " + args.choose_model)
      if args.choose_model == 'lenet' :
            PARAM['chosen_model'] = 'lenet'
      elif args.choose_model == 'nvidia' :
            PARAM['chosen_model'] = 'nvidia'
      else:
           print("Incorrect Parameter in the Model  ")
           PARAM['chosen_model'] = 'nvidia'
    
    if args.training_data:
       print("Training Data Path  " + args.training_data)
       if (os.path.exists(args.training_data + "/IMG")):
         PARAM['path2data'] = args.training_data
       else:
         PARAM['path2data'] = '/home/kputtur/Desktop/simulator'
         
    if args.epochs:
      print("Number of EPOCHS  " + str(args.epochs))
      PARAM['epoch'] = args.epochs
      
    if args.batchsize:
      print("Batchsize to train on "+ str(args.batchsize))
      PARAM['batchsize'] = args.batchsize
    
    if args.train_epochs:
      print("Number of train steps per epochs  " + str(args.train_epochs))
      PARAM['samples_per_epoch'] = args.train_epochs
      
    if args.recursive:
      print("Recursive Folders to append lines " + str(args.recursive))
      PARAM['recursive'] = args.recursive
 
lines = []
if args.recursive:
      dirs = os.listdir(PARAM['path2data'])
      for file in dirs:
        dirname = os.path.join(PARAM['path2data'], file)
        lines.extend(load_csv_file(dirname))
        
else:
        lines = load_csv_file(PARAM['path2data'])
     

print('Total number of Samples {}'.format(len(lines)))

#let's remove the zero lines
#nzero_lines = remove_zeros(lines)

#print('Total number of Samples {}'.format(len(nzero_lines)))

#let's shuffle before splitting
shuffle(lines)

#split the data 80% training 20% validation
train_data, validation_data = train_test_split(lines, test_size=0.2)

print('Starting Training Generator')
train_generator = data_generator(train_data, batch_size=PARAM['batchsize'])

print('Validation Generator')
validation_generator = data_generator(validation_data, batch_size=PARAM['batchsize'])


if (PARAM['chosen_model'] == 'nvidia'):
  nvidia_model = nvidia_model(summary=True)
  nvidia_model.compile(optimizer='adam', loss='mse')
 # train_steps_per_epoch = PARAM['samples_per_epoch']
  #validation_steps_per_epoch = len(validation_data)
  
  train_steps_per_epoch = 6 * PARAM['batchsize'] * int(len(train_data)/PARAM['batchsize'])
  validation_steps_per_epoch = 3 * PARAM['batchsize'] * int(len(validation_data)/PARAM['batchsize'])

  print('train_steps_per_epoch {}'.format(train_steps_per_epoch))
  print('validation_steps_per_epoch {}'.format(validation_steps_per_epoch))

  nvidia_model_history = nvidia_model.fit_generator(train_generator,
                        samples_per_epoch=train_steps_per_epoch,
                        validation_data = validation_generator,
                        nb_val_samples = validation_steps_per_epoch,
                        nb_epoch=PARAM['epoch'],
                        verbose=1)

  nvidia_model.save(PARAM['save_model_file'] + '.h5')
  with open(PARAM['save_model_file']+ '.json', 'w') as f:
    f.write(nvidia_model.to_json())
 
  plot_loss(model_history=nvidia_model_history)
  gc.collect()

if (PARAM['chosen_model'] == 'lenet'):
  lenet_model = LeNet_model(summary=True)
  lenet_model.compile(optimizer='adam', loss='mse')

#train_steps_per_epoch = 2 * PARAM['batchsize'] * int(len(train_data)/PARAM['batchsize'])
#validation_steps_per_epoch = PARAM['batchsize'] * int(len(validation_data)/PARAM['batchsize']) 

  train_steps_per_epoch = 6 * PARAM['bathsize'] * int(len(train_data)/PARAM['batchsize'])
  validation_steps_per_epoch = 3 * PARAM['bathsize'] * int(len(validation_data)/PARAM['batchsize'])

  print('train_steps_per_epoch {}'.format(train_steps_per_epoch))
  print('validation_steps_per_epoch {}'.format(validation_steps_per_epoch))

  lenet_model_history = lenet_model.fit_generator(train_generator,
            samples_per_epoch=train_steps_per_epoch,
            validation_data = validation_generator,
            nb_val_samples = validation_steps_per_epoch,
            nb_epoch=PARAM['epoch'],
            verbose=1)
  lenet_model.save(PARAM['save_model_file'] + '.h5')
  with open(PARAM['save_model_file']+ '.json', 'w') as f:
    f.write(lenet_model.to_json())
 
  plot_loss(model_history=lenet_model_history)
  gc.collect()



