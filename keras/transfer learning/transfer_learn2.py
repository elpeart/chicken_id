# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:58:22 2020

@author: elpea
"""

from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten
import keras
from keras.models import Model
import matplotlib.pyplot as plt

model = VGG16(include_top = False, input_shape = (224, 224, 3))
for layer in model.layers:
    layer.trainable = False
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