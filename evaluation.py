# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:51:14 2018

@author: murata
"""
import os, csv
import matplotlib
if os.name=='posix':
    matplotlib.use("agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from keras.utils.training_utils import multi_gpu_model
import keras
import seunet_model, train_main
from scipy.ndimage import label
from PIL import Image
import shutil


def load_trained_seunet(path_to_cnn,
                        epoch,
                        crop_shape,
                        nb_gpus,
                        ):
    path_to_model_weights = "weights_epoch=%03d.h5" % epoch
    filter_list_encoding = np.load(path_to_save_filter_list % "encoding")
    filter_list_decoding = np.load(path_to_save_filter_list % "decoding")
    
    img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
    model_single_gpu = seunet_model.seunet(img_dims, output_dims, filter_list_encoding, filter_list_decoding)
    model_single_gpu.load_weights(path_to_model_weights)
    if int(nb_gpus) > 1:
        model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
    else:
        model_multi_gpu = model_single_gpu
    return model_multi_gpu

def sensitivity_specificity(path_to_model_weights,
                            crop_shape=(64,64),
                            threshold=0.5,
                            batch_size=32,
                            nb_gpus=1,
                            ):
    path_to_validation_data = "../IntermediateData/validation_data.npy"
    path_to_validation_label = "../IntermediateData/validation_label.npy"
    data = np.load(path_to_validation_data)
    labels = np.load(path_to_validation_label)
    
    img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
    model_single_gpu = seunet_model.seunet(img_dims, output_dims)
    model_single_gpu.load_weights(path_to_model_weights)
    if int(nb_gpus) > 1:
        model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
    else:
        model_multi_gpu = model_single_gpu
    
    predicted = model_multi_gpu.predict(data, batch_size=batch_size)
    predicted[predicted>threshold] = 1
    predicted[predicted<=threshold] = 0
    
    sensitivity = predicted[(predicted==1) & (labels==1)].size / float(labels[labels==1].size)
    specificity = predicted[(predicted==0) & (labels==0)].size / float(labels[labels==0].size)

    return sensitivity, specificity


def object_level_dice_2d(y_true, y_pred): # y_true.shape = (画像のindex a, y, x)
    def dice_coeff(g ,s):
        return 2*(np.sum(g*s)+1) / (np.sum(g)+np.sum(s)+1)
    
    s_sum, g_tilde_sum = np.sum(y_pred), np.sum(y_true) # omega, omega_tilde の分子
    
    dice_object=0
    for a in range(len(y_true)):
        labeled_true, num_labels_true = label(y_true[a])
        labeled_pred, num_labels_pred = label(y_pred[a])
        
        # initialize
        g_tilde = np.zeros( (num_labels_true,)+y_true.shape[1:], dtype=np.uint8 )
        s = np.zeros( (num_labels_pred,)+y_true.shape[1:], dtype=np.uint8 )
        omega = np.zeros(num_labels_pred, dtype=np.uint8)
        omega_tilde = np.zeros(num_labels_true, dtype=np.uint8)
        # set g_tilde and s
        for i in range(num_labels_true):
            g_tilde[i][labeled_true==i+1] = 1
            omega_tilde[i] = np.sum(g_tilde[i]) / g_tilde_sum
        for i in range(num_labels_pred):
            s[i][labeled_pred==i+1] = 1
            omega[i] = np.sum(s[i]) / s_sum
        
        # compute Dice(G, S)
        dice_sg = np.zeros(num_labels_pred, dtype=np.uint8)
        for i in range(num_labels_pred):
            dice_sg[i] = 0
            for j in range(num_labels_true):
                dice_sg[i] = max( dice_sg[i], dice_coeff(g_tilde[j], s[i]) )
        # compute Dice(G_tilde, S_tilde)
        dice_sg_tilde = np.zeros(num_labels_true, dtype=np.uint8)
        for i in range(num_labels_true):
            dice_sg_tilde[i] = 0
            for j in range(num_labels_pred):
                dice_sg_tilde[i] = max( dice_sg_tilde[i], dice_coeff(g_tilde[i], s[j]) )
        
        dice_object += 0.5 * ( np.sum(omega*dice_sg) + np.sum(omega_tilde*dice_sg_tilde) )
    
    return dice_object


def whole_slide_dice_coeff(path_to_model_weights,
                           image_ids=np.arange(39,41),
                           data_shape=(584,565),
                           crop_shape=(64,64),
                           nb_gpus=1,
                           ):
#    def dice_coeff(g ,s):
#        return 2*(np.sum(g*s)+1) / (np.sum(g)+np.sum(s)+1)
    def load_model(path_to_model_weights):
        img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
        model_single_gpu = seunet_model.seunet(img_dims, output_dims)
        model_single_gpu.load_weights(path_to_model_weights)
        if int(nb_gpus) > 1:
            model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
        else:
            model_multi_gpu = model_single_gpu
        return model_multi_gpu
    
    model_multi_gpu = load_model(path_to_model_weights)
    
    images, manuals = train_main.load_image_manual(image_ids=image_ids,
                                                   data_shape=data_shape,
                                                   )
    def dice_coeff_wsi(count_image):
        count = 0
#        data_size = (1+data_shape[0]//crop_shape[0]) * (1+data_shape[1]//crop_shape[1])
        data_size = (data_shape[0]//crop_shape[0]) * (data_shape[1]//crop_shape[1])
#        print(data_size)
        data = np.zeros( (data_size,)+crop_shape+(3,), dtype=np.uint8 )
        labeled = np.zeros( (data_size,)+crop_shape+(1,), dtype=np.uint8 )
        for y in range(0, data_shape[0]-crop_shape[0], crop_shape[0]):
            for x in range(0, data_shape[1]-crop_shape[1], crop_shape[1]):
                data[count] = images[count_image, y:y+crop_shape[0], x:x+crop_shape[1],:]
                labeled[count] = manuals[count_image, y:y+crop_shape[0], x:x+crop_shape[1],:]
                count += 1
        predicted = np.round( model_multi_gpu.predict(data, batch_size=32) )
        sum_groundtruth = np.sum(manuals[count_image])
        sum_predict = np.sum(predicted)
        dice_numerator = 0
        for count in range(data_size):
            dice_numerator += 2 * np.sum( labeled[count] * predicted[count] )
        return dice_numerator / (sum_groundtruth+sum_predict)
    
    dice_sum=0
    for count_image in range(len(image_ids)):
        dice_sum += dice_coeff_wsi(count_image)
    
    return dice_sum / len(image_ids)
    

#def whole_slide_accuracy(path_to_model_weights,
#                         image_ids=np.arange(39,41),
#                         data_shape=(584,565),
#                         crop_shape=(64,64),
#                         nb_gpus=1,
#                         ):
#    path_to_mask = "../training/mask/%d_training_mask.gif" # % image_id
#    
#    def load_model(path_to_model_weights):
#        img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
#        model_single_gpu = seunet_model.seunet(img_dims, output_dims)
#        model_single_gpu.load_weights(path_to_model_weights)
#        if int(nb_gpus) > 1:
#            model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
#        else:
#            model_multi_gpu = model_single_gpu
#        return model_multi_gpu
#    
#    model_multi_gpu = load_model(path_to_model_weights)
#    
#    images, manuals = train_main.load_image_manual(image_ids=image_ids,
#                                                   data_shape=data_shape,
#                                                   )
#    
#    pixel_sum, true_sum = 0,0
##    mask = np.zeros( (image_ids.shape+data_shape+(1,)), dtype=np.uint8 )
#    for count_image in range(image_ids.size):
#        image_id = image_ids[count_image]
#        mask = np.array( Image.open(path_to_mask % (image_id)) )
#        mask = mask / np.amax(mask)
#        mask = mask.reshape(mask.shape+(1,))
#        
#        data_size = (data_shape[0]//crop_shape[0]) * (data_shape[1]//crop_shape[1])
#        data = np.zeros( (data_size,)+crop_shape+(3,), dtype=np.uint8 )
#        labeled = np.zeros( (data_size,)+crop_shape+(1,), dtype=np.uint8 )
#        masked = np.zeros( (data_size,)+crop_shape+(1,), dtype=np.uint8 )
#        count = 0
#        for y in range(0, data_shape[0]-crop_shape[0], crop_shape[0]):
#            for x in range(0, data_shape[1]-crop_shape[1], crop_shape[1]):
#                data[count] = images[count_image, y:y+crop_shape[0], x:x+crop_shape[1],:]
#                labeled[count] = manuals[count_image, y:y+crop_shape[0], x:x+crop_shape[1],:]
#                masked[count] = mask[y:y+crop_shape[0], x:x+crop_shape[1],:]
#                count += 1
#        predicted = np.round( model_multi_gpu.predict(data, batch_size=32) )
#        pixel_sum += float( mask[mask>0].size )
#        true_sum=0
#        for count in range(data_size):
#            true_sum += np.sum(masked[count]*labeled[count]*predicted[count] + masked[count]*(1-labeled[count])*(1-predicted[count]))
##            labeled[count][labeled[count]==1 & predicted[count]==1].size
#           
#    return true_sum / float(pixel_sum)
        

def whole_slide_prediction(path_to_cnn,
                           epoch,
                           path_to_model_weights="",
                           model="",
                           image_id=38,
                           data_shape=(584,565),
                           crop_shape=(64,64),
                           nb_gpus=1,
                           dataset="train",
                           if_save_img=True,
                           if_save_npy=False,
                           batch_size=32,
                           ):
    if path_to_model_weights=="":
        path_to_model_weights = path_to_cnn+"weights_epoch=%03d.h5" % epoch
    if dataset=="train":
        path_to_mask = "../training/mask/%d_training_mask.gif" # % image_id
        path_to_image = "../training/images/%d_training.tif" # % image_id
    if dataset=="test":
        path_to_mask = "../test/mask/%02d_test_mask.gif" # % image_id
        path_to_image = "../test/images/%02d_test.tif" # % image_id
        
        
    if model=="":
        model = load_trained_seunet(path_to_cnn, epoch, crop_shape, nb_gpus)
    
    image = np.array( Image.open(path_to_image % (image_id)) )
    mask = np.array( Image.open(path_to_mask % (image_id)) )
    mask = mask / np.amax(mask)
    
    data_size = int( np.ceil(data_shape[0]/float(crop_shape[0])) * np.ceil(data_shape[1]/float(crop_shape[1])) )
    data = np.zeros( (data_size,)+crop_shape+(3,), dtype=np.uint8 )
    count = 0
    for y in range(0, data_shape[0], crop_shape[0]):
        if y+crop_shape[0] > data_shape[0]:
            y = data_shape[0]-crop_shape[0]
        for x in range(0, data_shape[1], crop_shape[1]):
            if x+crop_shape[1] > data_shape[1]:
                x = data_shape[1]-crop_shape[1]
            data[count] = image[y:y+crop_shape[0], x:x+crop_shape[1],:]
            count += 1
#    crop_predicted = np.round( model.predict(data, batch_size=batch_size) )
    crop_predicted = model.predict(data, batch_size=batch_size)
    
    whole_slide_predicted = np.zeros(image.shape[:-1], dtype=np.float32)
    count = 0
    for y in range(0, data_shape[0], crop_shape[0]):
        if y+crop_shape[0] > data_shape[0]:
            y = data_shape[0]-crop_shape[0]
        for x in range(0, data_shape[1], crop_shape[1]):
            if x+crop_shape[1] > data_shape[1]:
                x = data_shape[1]-crop_shape[1]
            whole_slide_predicted[y:y+crop_shape[0], x:x+crop_shape[1]] = crop_predicted[count].reshape(crop_shape)
            count += 1
    
    whole_slide_predicted = whole_slide_predicted*mask
    
    # png 形式で保存
    if if_save_img:
        if not os.path.exists(path_to_model_weights[:-3]+'/'):
            os.makedirs(path_to_model_weights[:-3]+'/')
        plt.imshow(whole_slide_predicted)
        plt.savefig(path_to_model_weights[:-3]+'/'+str(image_id)+'.png')
    
    # numpy 形式で保存
    if if_save_npy:
        np.save(path_to_cnn+"%02d.npy" % (image_id), whole_slide_predicted)
            
    return whole_slide_predicted

def whole_slide_accuracy(path_to_cnn,
                         epoch,
                         path_to_model_weights="",
                         model="",
                         image_ids=[],
                         data_shape=(584,565),
                         crop_shape=(64,64),
                         if_save_img = True,
                         nb_gpus=1,
                         batch_size=32,
                         ):
#    path_to_model_weights = "weights_epoch=%03d.h5" % epoch
    path_to_train_manual = "../training/1st_manual/%d_manual1.gif" # % image_id
    path_to_mask = "../training/mask/%d_training_mask.gif" # % image_id
    
    accuracy = np.zeros(len(image_ids))
    for _id in range(len(image_ids)):
        image_id = image_ids[_id]
        #load prediction
        prediction = whole_slide_prediction(path_to_cnn=path_to_cnn,
                                            epoch=epoch,
                                            path_to_model_weights=path_to_model_weights,
                                            model=model,
                                            image_id=image_id,
                                            data_shape=data_shape,
                                            crop_shape=crop_shape,
                                            nb_gpus=nb_gpus,
                                            if_save_img=if_save_img,
                                            batch_size=batch_size,
                                            )    
        #load ground truth
        manual = np.array( Image.open(path_to_train_manual % (image_id)) )
        manual[manual>0]=1
        mask = np.array( Image.open(path_to_mask % (image_id)) )
        mask[mask>0]=1
        
        
        # しきい値処理
        prediction[prediction>=0.5]=1
        prediction[prediction<0.5]=0
        
        accuracy[_id] = prediction[(prediction==manual) & (mask>0)].size / float( manual[mask>0].size )
    
    accuracy_average = accuracy.sum() / float(accuracy.size)
    
    return accuracy_average


#def test_accuracy(path_to_gt_dir="../test/1st_manual/",
#                  path_to_predict_dir="",
#                  threshold=0.5,
#                  ):
#    path_to_mask = "../test/mask/%02d_test_mask.gif" # % image_id
#    
#    accuracy=0
#    for image_id in range(1,21):
#        #load ground truth
#        manual = np.array( Image.open(path_to_gt_dir + "%02d_manual1.gif" % (image_id)) )
#        manual[manual>0]=1
#        mask = np.array( Image.open(path_to_mask % (image_id)) )
#        mask[mask>0]=1
#        
#        prediction = np.load(path_to_predict_dir + "%02d.npy" % image_id)
#         # しきい値処理
#        prediction[prediction>=threshold]=1
#        prediction[prediction<threshold]=0
#
#        accuracy += prediction[(prediction==manual) & (mask>0)].size / float( manual[mask>0].size )
#      
#    return accuracy / 20.0

def group_accuracies(group="validation",
                     path_to_predict_dir="",
                     image_ids=[],
                     thresholds=[0.5,0.6,7,0.8,0.9,0.99,0.999],
                     ):
    path_to_csv = path_to_predict_dir + "%s_accuracies.csv" % group
    if group=="test":
        path_to_gt_dir="../test/1st_manual/"
        path_to_mask = "../test/mask/%02d_test_mask.gif" # % image_id
    elif group=="validation":
        path_to_gt_dir="../training/1st_manual/"
        path_to_mask = "../training/mask/%02d_training_mask.gif" # % image_id
        
    group_acc_csv = open(path_to_csv, 'w')
    writer = csv.writer(group_acc_csv, lineterminator='\n') 
    writer.writerow(["threshold", "accuracy",]) # headder
    
    for threshold in thresholds:
        accuracy=0
        for image_id in image_ids:
            #load ground truth
            manual = np.array( Image.open(path_to_gt_dir + "%02d_manual1.gif" % (image_id)) )
            manual[manual>0]=1
            mask = np.array( Image.open(path_to_mask % (image_id)) )
            mask[mask>0]=1
            
            prediction = np.load(path_to_predict_dir + "%02d.npy" % image_id)
             # しきい値処理
            prediction[prediction>=threshold]=1
            prediction[prediction<threshold]=0
    
            accuracy += prediction[(prediction==manual) & (mask>0)].size / float( manual[mask>0].size )
        accuracy = accuracy / float(len(image_ids))
        writer.writerow([threshold, accuracy])
        
    group_acc_csv.close()


def accuracy_tops(path_to_cnn,
                  ):
    path_to_csv = path_to_cnn + "test_acc_maxs.csv"
    acc_csv = open(path_to_csv, 'w')
    writer = csv.writer(acc_csv, lineterminator='\n') 
    writer.writerow(["validation_max", "test_acc_at_valmax","threshold"]) # headder
    path_to_group_csv = path_to_cnn + "%s/%s_accuracies.csv" # % (model, group)
    for model in os.listdir(path_to_cnn):
        if os.path.exists(path_to_group_csv % (model, "validation")):
            validation_accs = pd.read_csv(path_to_group_csv % (model, "validation")).values
            test_accs = pd.read_csv(path_to_group_csv % (model, "test")).values
            max_index =np.argmax(validation_accs[:,1])
            threshold = validation_accs[max_index,0]
            val_max = validation_accs[max_index,1]
            test_acc_at_valmax = test_accs[max_index,1]
            writer.writerow([val_max, test_acc_at_valmax, threshold])
    acc_csv.close()
    
    
def main():
#    image_id=39
    path_to_cnn = "../output/mm05dd10_02/"
    accuracy_tops(path_to_cnn=path_to_cnn)
#    epoch = 64
#    dice_coeff = whole_slide_dice_coeff(path_to_model_weights,
##                                        image_ids=np.arange(18,20),
#                                        data_shape=(584,565),
#                                        crop_shape=(64,64),
#                                        nb_gpus=1,
#                                        )
#    print(dice_coeff)
#    acc = whole_slide_accuracy(path_to_model_weights,
#                               image_ids=np.arange(39,41),
#                               data_shape=(584,565),
#                               crop_shape=(64,64),
#                               nb_gpus=1,
#                               )
#    print(acc)    
#    whole_slide_predicted = whole_slide_prediction(path_to_model_cnn=path_to_model_cnn,
#                                                   epoch=epoch,
#                                                   image_id=image_id,
#                                                   data_shape=(584,565),
#                                                   crop_shape=(32,32),
#                                                   nb_gpus=1,
#                                                   batch_size=32,
#                                                   )
#    print(whole_slide_predicted.shape)
#    print(np.unique(whole_slide_predicted))
#    if not os.path.exists(path_to_model_weights[:-3]+'/'):
#        os.makedirs(path_to_model_weights[:-3]+'/')
#    plt.imshow(whole_slide_predicted)
#    plt.savefig(path_to_model_weights[:-3]+'/'+str(image_id)+'.png')
#    Image.fromarray(whole_slide_predicted).save(path_to_model_weights[:-3]+str(image_id)+'.gif')
#    sensitivity, specificity = sensitivity_specificity(path_to_model_weights,
#                                                       crop_shape=(64,64),
#                                                       threshold=0.5,
#                                                       batch_size=32,
#                                                       nb_gpus=1,
#                                                       )    
#    print(sensitivity, specificity)
    
    
if __name__ == '__main__':
    main()
