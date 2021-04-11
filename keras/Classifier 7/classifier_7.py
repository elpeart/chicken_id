# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:49:56 2020

@author: elpea
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.utils import shuffle
import os
import cv2 
DATADIR ="C:\\Users\\elpea\\OneDrive - Old Dominion University\\ECE 607 Final Project"
CATEGORIES = ["Buff Orpington","Rhode Island Red","Silver Laced Wyandotte","White Leghorn"]
# Define image size for all data images input
img_size = 500
# create training data 
# function to setup training data
#By ussing cv2 package we are already importing images in np.array format 
training_data = [] #define training data array 
def create_training_data ():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category) 
        chicken_class_num = CATEGORIES.index(category) # assigns a index number to the chicken class
        for img in os.listdir(path):
        # Create an exception in case the images are broken
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(img_size,img_size))#resize all images to 100 x100 pixels
                training_data.append([new_array,chicken_class_num])
            except  Exception as e:
                pass
create_training_data()
 
# Print length of the array training_data to make sure is taking all images from defined directory        
print(len(training_data))        
# Shuffle the training data  
training_data = shuffle(training_data)

# Initialize x and y inputs for model
x = []
y = []
# Split into an array of data and an array of labels
for i in training_data:
    x.append(i[0])
    y.append(i[1])

y = keras.utils.to_categorical(y, 4)

x = np.array(x)
y = np.array(y)

# Format the data
if K.image_data_format() == 'channels_first':
    x = x.reshape(x.shape[0], 1, img_size, img_size)
    input_shape = (1, img_size, img_size)
else:
    x = x.reshape(x.shape[0], img_size, img_size, 1)
    input_shape = (img_size, img_size, 1)

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dense(4, activation = 'softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('model_{epoch:02d}_{val_accuracy:.4f}.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Fit the data
history = model.fit(x, y, validation_split = 0.2, epochs = 50, batch_size = 10, callbacks = [checkpoint])
