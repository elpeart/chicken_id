# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:32:45 2020

@author: elpea
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:00:39 2020

@author: elpea
"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten
import keras
from keras.models import Model
from matplotlib.pyplot import plot as plt
import numpy as np
import os
from sklearn.utils import shuffle

DATADIR ="C:\\Users\\elpea\\OneDrive - Old Dominion University\\ECE 607 Final Project"
CATEGORIES = ["Buff Orpington","Rhode Island Red","Silver Laced Wyandotte","White Leghorn"]


training_data = []
for category in CATEGORIES:
    path = os.path.join(DATADIR,category) 
    chicken_class_num = CATEGORIES.index(category) # assigns a index number to the chicken class
    for img in os.listdir(path):
        new_img = load_img(os.path.join(path, img), target_size = (224,224))
        new_array = img_to_array(new_img)
        training_data.append([new_array, chicken_class_num])

training_data = shuffle(training_data)

# Initialize x and y inputs for model
x = []
y = []
# Split into an array of data and an array of labels
for i in training_data:
    x.append(i[0])
    y.append(i[1])

for img in x:
    img = preprocess_input(img)
y = to_categorical(y, 4)

x = np.array(x)
y = np.array(y)

model = VGG16(include_top = False, input_shape = (224, 224, 3))

# add new classifier layers
flat1 = Flatten()(model.outputs)
class1 = Dense(128, activation='relu')(flat1)
output = Dense(4, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('model_{epoch:02d}_{val_accuracy:.4f}.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Fit the data
history = model.fit(x, y, validation_split = 0.2, epochs = 60, batch_size = 1, callbacks = [checkpoint])

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()