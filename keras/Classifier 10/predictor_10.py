# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:22:56 2020

@author: elpea
"""

from keras.models import load_model
import numpy as np
import cv2 


model = load_model('model_34_0.9720.h5')

img_size = 500
test_image = cv2.imread('IMG_3143.JPG',cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image,(img_size,img_size))

test_image = test_image.reshape(1, img_size, img_size, 1)

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