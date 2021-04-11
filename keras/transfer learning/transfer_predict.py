# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:58:24 2020

@author: elpea
"""

from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input


model = load_model('model_17_0.9903.h5')

test_image = load_img('IMG-8317.jpg', target_size = (224, 224))
test_image = img_to_array(test_image)
test_image = preprocess_input(test_image)

test_image = test_image.reshape(1, test_image.shape[0], test_image.shape[1], test_image.shape[2])

#predict the result
result = model.predict(test_image)
CATEGORIES = ["Buff Orpington","Rhode Island Red","Silver Laced Wyandotte","White Leghorn"]
result = list(result[0,:])
cat = np.argmax(result)
print(CATEGORIES[cat], ':', 100 * float(result[cat]))
del CATEGORIES[cat]
del result[cat]
cat2 = np.argmax(result)
print(CATEGORIES[cat2], ':', 100*float(result[cat2]))
del CATEGORIES[cat2]
del result[cat2]
cat3 = np.argmax(result)
print(CATEGORIES[cat3], ':', 100*float(result[cat3]))