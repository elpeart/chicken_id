# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:53:33 2020

@author: elpea
"""

 ####### program to import data images from directory #######
import os
import cv2 
from sklearn.utils import shuffle
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