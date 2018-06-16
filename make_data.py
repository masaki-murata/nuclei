# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:23:25 2018

@author: murata
"""

import numpy as np
from PIL import Image, ImageOps

def load_data():
    path_to_train_image = "../training/images/%d_training.tif"
    for image_id in range(21, 41):
        image = np.array( Image.open(path_to_train_image % image_id) )
        print(image.shape)
        

def make_dataset(data_size=2**22,
                 image_ids=list(range(20)),
                 data_shape=(584,565),
                 crop_shape=(64,64)
                 ):
    path_to_train_image = "../training/images/%d_training.tif" # % image_id
    path_to_train_manual = "../training/1st_manual/%d_manual1.gif" # % image_id
    path_to_data = "../IntermediateData/data.npy"
    path_to_label = "../IntermediateData/label.npy"
    
    
    images = np.zeros( ((len(image_ids),)+data_shape+(3,)), dtype=np.uint8 )
    manuals = np.zeros( ((len(image_ids),)+data_shape+(1,)), dtype=np.uint8 )
    for image_id in image_ids:
        images[image_id] = np.array( Image.open(path_to_train_image % (image_id+21)) )
        manual = np.array( Image.open(path_to_train_manual % (image_id+21)) )
        manuals[image_id] = manual.reshape(manual.shape+(1,))
    
    data = np.zeros( ((data_size,)+crop_shape+(3,)), dtype=np.uint8 )
    label = np.zeros( ((data_size,)+crop_shape+(1,)), dtype=np.uint8 )
    for count in range(data_size):
        print(count,",", end="")
        image_id = np.random.choice(image_ids)
        y = np.random.randint(data_shape[0]-crop_shape[0])
        x = np.random.randint(data_shape[1]-crop_shape[1])
#        crop_pos=(y,x)
        data[count] = images[image_id][y:y+crop_shape[0], x:x+crop_shape[1]]
        label[count] = manuals[image_id][y:y+crop_shape[0], x:x+crop_shape[1]]
    np.save(path_to_data, data)
    np.save(path_to_label, label)
    
make_dataset()