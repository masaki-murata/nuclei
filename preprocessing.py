# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:33:49 2018

@author: murata
"""

import readmhd
import os, re, csv, sys
import attention_senet_based
from skimage.transform import resize
import numpy as np
from seunet_model import seunet
#import keras
from PIL import Image
#import matplotlib.pyplot as plt


# 画像のサイズを統一する
def resize_all(ref_size=np.array([3.6, 0.625, 0.625])):
    path_to_dir = "../TrainingData_Part%d/"
    path_to_original_scale_npy = "../IntermediateData/original_scale/%s.npy"
    path_to_rescaled_npy = "../IntermediateData/rescaled/%s.npy"
    for part in range(1,4):
        for file in os.listdir(path_to_dir % part):
            if re.match("Case[0-9][0-9].mhd", file):
                volume = readmhd.read(path_to_dir % part + file)
                np.save(path_to_original_scale_npy % file[:-4], np.float32(volume.vol))
                matrixsize_rescaled = (volume.matrixsize[::-1]*(volume.voxelsize[::-1] / ref_size)).astype(np.int)
                matrixsize_rescaled[1]=matrixsize_rescaled[2]
                volume_rescaled = resize(volume.vol, matrixsize_rescaled)
#                np.float32(volume_rescaled).tofile("../IntermediateData/rescaled/"+file[:-4]+".raw")
                np.save(path_to_rescaled_npy % file[:-4], np.float32(volume_rescaled))
                print(file, volume_rescaled.shape)
            if re.match("Case[0-9][0-9]_segmentation.mhd", file):
                volume = readmhd.read(path_to_dir % part + file)
                np.save(path_to_original_scale_npy % (file[:-4]+"32"), np.float32(volume.vol))
                matrixsize_rescaled = (volume.matrixsize[::-1]*(volume.voxelsize[::-1] / ref_size)).astype(np.int)
                matrixsize_rescaled[1]=matrixsize_rescaled[2]
                volume_rescaled = resize(np.float32(volume.vol), matrixsize_rescaled)
#                volume_rescaled.tofile("../IntermediateData/rescaled/"+file[:-4]+".raw")
                volume_rescaled[volume_rescaled>0.5] = 1
                np.save(path_to_rescaled_npy % file[:-4], np.int8(volume_rescaled))
                np.save(path_to_rescaled_npy % (file[:-4]+"32"), np.float32(volume_rescaled))
                print(file, volume_rescaled.shape)
