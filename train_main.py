#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:22:59 2018

@author: muratamasaki
"""

import numpy as np
from PIL import Image
import os, datetime, math, re, csv
import seunet_model, seunet_main, evaluation
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.utils.training_utils import multi_gpu_model
import keras.backend as K
import shutil
import matplotlib.pyplot as plt


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
if os.name=='posix':
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="2,3", # specify GPU number
            allow_growth=True
        )
    )
    
    set_session(tf.Session(config=config))


# 画像と正解ラベルを読み込む関数
def load_image_groundtruths(image_ids=np.arange(1,15),
#                      data_shape=(584,565),
#                      crop_shape=(64,64),
                      ):
    path_to_train_image = "../segmentation_training_set/image%02d.png" # % image_id
    path_to_train_mask = "../segmentation_training_set/image%02d_mask.txt" # % image_id

    def load_grountruth(image_id):
        f = open(path_to_train_mask % image_id)
        line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)
        shape = list(map(int, line.split())) # 第一行は shape
        
        grountruth = np.zeros(shape[0]*shape[1])
        count = 0
        line = f.readline()
        while line:
            grountruth[count] = line
            count += 1
            line = f.readline()
        f.close
        grountruth = grountruth.reshape((shape[1],shape[0]))
#        print(grountruth[0,0], np.amax(grountruth))
        grountruth[grountruth>0] = 255
#        Image.fromarray(grountruth).show()
        return grountruth
        
    # load data
    images, groundtruths = {}, {}
    for x in range(image_ids.size):
        image_id = image_ids[x]
        images[str(image_id)] = np.array( Image.open(path_to_train_image % (image_id)) )
        groundtruth = np.array( Image.open(path_to_train_mask % (image_id)) )
        groundtruth[groundtruth>0] = 1
        groundtruths[str(image_id)] = groundtruth.reshape(groundtruth.shape+(1,))
        
    return images, groundtruths

    
def make_validation_dataset(validation_ids=np.arange(14,15),
                            load = True,
                            val_data_size = 1024,
#                            data_shape=(584,565),
                            crop_shape=(128,128),
                            ):
    path_to_validation_data = "../IntermediateData/validation_data_crop%d%d.npy" % (crop_shape[0], crop_shape[1])
    path_to_validation_label = "../IntermediateData/validation_label_crop%d%d.npy" % (crop_shape[0], crop_shape[1])
    if load==True and os.path.exists(path_to_validation_data) and os.path.exists(path_to_validation_label):
        data = np.load(path_to_validation_data)
        labels = np.load(path_to_validation_label)
    else:
        images, groundtruths = load_image_groundtruths(image_ids=validation_ids,
#                                            data_shape=data_shape,
#                                            crop_shape=crop_shape,
                                            )
        data = np.zeros( (val_data_size,)+crop_shape+(3,), dtype=np.uint8 )
        labels = np.zeros( (val_data_size,)+crop_shape+(1,), dtype=np.uint8 )
        for count in range(val_data_size):
            image_num = np.random.randint(images.shape[0])
            image, groundtruth = images[str(image_num)], groundtruths[str(image_num)]
            theta = np.random.randint(360)
            (h, w) = crop_shape # w は横、h は縦
            c, s = np.abs(np.cos(np.deg2rad(theta))), np.abs(np.sin(np.deg2rad(theta)))
            (H, W) = (int(s*w + c*h), int(c*w + s*h)) #最終的に切り出したい画像に内接する四角形の辺の長さ
            y, x = np.random.randint(images.shape[1] - H + 1), np.random.randint(images.shape[2] - W + 1) # 第一段階での左上の座標
#            y = np.random.randint(images.shape[1]-crop_shape[0])
#            x = np.random.randint(images.shape[2]-crop_shape[1])
            data_crop, label_crop = Image.fromarray(image[y:y+H, x:x+W,:]), Image.fromarray(groundtruth[y:y+H, x:x+W])
            data_crop, label_crop = np.array(data_crop.rotate(-theta, expand=True)), np.array(label_crop.rotate(-theta, expand=True))
            y_min, x_min = data_crop.shape[0]//2-h//2, data_crop.shape[1]//2-w//2
            data_crop, label_crop = data_crop[y_min:y_min+h, x_min:x_min+w,:], label_crop[y_min:y_min+h, x_min:x_min+w]
            label_crop = label_crop.reshape(label_crop.shape+(1,))
            if np.random.choice([True,False]):
                data_crop, label_crop = np.flip(data_crop, axis=1), np.flip(label_crop, axis=1)
#                if np.random.choice([True,False]):
#                    data_crop, label_crop = np.flip(data_crop, axis=2), np.flip(label_crop, axis=2)
            data[count], labels[count] = data_crop, label_crop
#            data[count] = images[image_num, y:y+crop_shape[0], x:x+crop_shape[1],:]
#            labels[count] = groundtruth[y:y+crop_shape[0], x:x+crop_shape[1],:]
        np.save(path_to_validation_data, data)
        np.save(path_to_validation_label, labels)
                
    return data, labels        


def batch_iter(images={}, # {画像数id、W, H, 3)}
               groundtruth={}, # {画像数id、W, H, 3)}
               crop_shape=(128,128),
               steps_per_epoch=2**14,
#               image_ids=np.arange(20),
               batch_size=32,
               ):
        
    groundtruth = groundtruth.reshape(groundtruth.shape[:-1])
    while True:
        for step in range(steps_per_epoch):
            data = np.zeros( (batch_size,)+crop_shape+(3,), dtype=np.uint8 )
            labels = np.zeros( (batch_size,)+crop_shape+(1,), dtype=np.uint8 )
            for count in range(batch_size):
                image_num = np.random.randint(images.shape[0])
                theta = np.random.randint(360)
                (h, w) = crop_shape # w は横、h は縦
                c, s = np.abs(np.cos(np.deg2rad(theta))), np.abs(np.sin(np.deg2rad(theta)))
                (H, W) = (int(s*w + c*h), int(c*w + s*h)) #最終的に切り出したい画像に内接する四角形の辺の長さ
                y, x = np.random.randint(images.shape[1] - H + 1), np.random.randint(images.shape[2] - W + 1) # 第一段階での左上の座標
#                y = np.random.randint(images.shape[1]-crop_shape[0])
#                x = np.random.randint(images.shape[2]-crop_shape[1])
#                data[count] = images[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]
#                label[count] = manuals[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]
#                data_crop = images[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]       
#                label_crop = manuals[image_id, y:y+crop_shape[0], x:x+crop_shape[1],:]
                data_crop, label_crop = Image.fromarray(images[image_num, y:y+H, x:x+W,:]), Image.fromarray(groundtruth[image_num, y:y+H, x:x+W])
                data_crop, label_crop = np.array(data_crop.rotate(-theta, expand=True)), np.array(label_crop.rotate(-theta, expand=True))
                y_min, x_min = data_crop.shape[0]//2-h//2, data_crop.shape[1]//2-w//2
                data_crop, label_crop = data_crop[y_min:y_min+h, x_min:x_min+w,:], label_crop[y_min:y_min+h, x_min:x_min+w]
                label_crop = label_crop.reshape(label_crop.shape+(1,))
                if np.random.choice([True,False]):
                    data_crop, label_crop = np.flip(data_crop, axis=1), np.flip(label_crop, axis=1)
#                if np.random.choice([True,False]):
#                    data_crop, label_crop = np.flip(data_crop, axis=2), np.flip(label_crop, axis=2)
                data[count], labels[count] = data_crop, label_crop
            yield data, labels
            

def train(train_ids=np.arange(21,39),
          validation_ids=np.arange(39,41),
          val_data_size = 2048,
          batch_size=32,
          data_size_per_epoch=2**14,
#          steps_per_epoch=2**14,
          epochs=256,
          data_shape=(584,565),
          crop_shape=(128,128),
          filter_list_encoding=np.array([]),
          filter_list_decoding=np.array([]),
          if_save_img=True,
          nb_gpus=1,
          ):
    
    steps_per_epoch=data_size_per_epoch//batch_size
    # set our model
    img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
    model_single_gpu = seunet_model.seunet(img_dims, output_dims, filter_list_encoding, filter_list_decoding)
    print(nb_gpus)
    if int(nb_gpus) > 1:
        model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
    else:
        model_multi_gpu = model_single_gpu

    
    # load data
    train_images, train_manuals = load_image_manual(image_ids=train_ids,data_shape=data_shape)
#    validation_images, validation_manuals = \
#        load_image_manual(image_ids=validation_ids,data_shape=data_shape,crop_shape=crop_shape)
    val_data, val_label = make_validation_dataset(validation_ids=validation_ids,
                                                  load = True,
                                                  val_data_size = val_data_size,
                                                  data_shape=data_shape,
                                                  crop_shape=crop_shape,
                                                  )
        
    train_gen = batch_iter(images=train_images,
                           manuals=train_manuals, 
                           crop_shape=crop_shape,
                           steps_per_epoch=steps_per_epoch,
                           batch_size=batch_size,
                           )

#    path_to_save_model = "../output/ep{epoch:04d}-valloss{val_loss:.4f}.h5"
    path_to_cnn_format = "../output/mm%02ddd%02d_%02d/"
    # make a folder to save history and models
    now = datetime.datetime.now()
    for count in range(10):
        path_to_cnn = path_to_cnn_format % (now.month, now.day, count)
        if not os.path.exists(path_to_cnn):
            os.mkdir(path_to_cnn)
            break
    path_to_code = "./train_main.py"
    path_to_code_moved = path_to_cnn + "train_main_used.py"
    path_to_save_model = path_to_cnn + "model_epoch=%03d.h5"
    path_to_save_weights = path_to_cnn + "weights_epoch=%03d.h5"
    path_to_save_filter_list = path_to_cnn + "filter_list_%s.npy" # % encoding or decoding
    
    shutil.copyfile(path_to_code, path_to_code_moved)
    
    np.save(path_to_save_filter_list % "encoding", filter_list_encoding)
    np.save(path_to_save_filter_list % "decoding", filter_list_decoding)
    
#    callbacks = []
#    callbacks.append(ModelCheckpoint(path_to_save_model, monitor='val_loss', save_best_only=False))
#    callbacks.append(CSVLogger("log%03d.csv" % counter))
#    callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0001 , patience=patience))
    opt_generator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    model_multi_gpu.compile(loss='binary_crossentropy', optimizer=opt_generator)
    model_multi_gpu.compile(loss=seunet_main.mean_dice_coef_loss, optimizer=opt_generator)
    
    for epoch in range(1,epochs+1):
        model_multi_gpu.fit_generator(train_gen,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=1,
#                                      epochs=epochs,
#                                      callbacks=callbacks,
                                      validation_data=(val_data,val_label)
                                      )
        print('Epoch %s/%s done' % (epoch, epochs))
        print("")
        
        if epoch>0 and epoch % 1==0:
            print(epoch)
            model_single_gpu.save(path_to_save_model % (epoch))
            model_single_gpu.save_weights(path_to_save_weights % (epoch))
            validation_accuracy = evaluation.whole_slide_accuracy(path_to_cnn=path_to_cnn,
                                                                  epoch=epoch,
                                                                  model=model_multi_gpu,
                                                                  image_ids=validation_ids,
                                                                  data_shape=data_shape,
                                                                  crop_shape=crop_shape,
                                                                  if_save_img=if_save_img,
                                                                  nb_gpus=nb_gpus,
                                                                  batch_size=batch_size,
                                                                  )
            print("validation_accuracy = ", validation_accuracy)

def dict_hyperparam():
    hp = {}
    hp["learning_rate"] = list(range(1,7))
#    hp["momentum"] = [0, 0.99]
#    hp["optimizer"] = ["SGD", "Adam"]
    hp["loss"] = ["binary_crossentropy", "mean_dice_coef_loss"]
    hp["batch_size"] = [2**x for x in range(3,6)] #[2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11] #[2**x for x in range(6)]
    hp["crop_shape"] = [2**x for x in range(4,10)]    
    
    conv_num_max = 8
    hp["conv_num"] = list(range(4,conv_num_max+1))
    for conv_id in range(1,conv_num_max+1):
        hp["filter_num_conv%d" % conv_id] = [2**x for x in range(4,8)]
        hp["filter_num_deconv%d" % conv_id] = [2**x for x in range(4,8)]

    return hp

# ハイパーパラメータをランダムに選択
def chose_hyperparam():
    hp = dict_hyperparam()
    hp_value = {}
    for hyperparam in hp.keys():
        rp = np.random.rand()
        index = int( math.floor( rp*len(hp[hyperparam]) ) )
        hp_value[hyperparam] = hp[hyperparam][index]
#        if hyperparam == "conv_layer_num" and hp_value[hyperparam]==2:            
#                hp["dense_units1"] = list(range(2,(voi_width//2//2)))
            
    
    hp_value["learning_rate"] = 10**(-hp_value["learning_rate"] )
    
    for x in range(1, hp_value["conv_num"]):
        if hp_value["filter_num_conv%d" % (x+1)] < hp_value["filter_num_conv%d" % x]:
            hp_value["fnfilter_num_conv%d" % (x+1)] = hp_value["filter_num_conv%d" % x]
        if hp_value["filter_num_deconv%d" % (x+1)] > hp_value["filter_num_deconv%d" % x]:
            hp_value["fnfilter_num_deconv%d" % (x+1)] = hp_value["filter_num_deconv%d" % x]
    hp_value["filter_list_encoding"] = [hp_value["filter_num_conv%d" % x] for x in range(1, hp_value["conv_num"]+1)]
    hp_value["filter_list_decodng"] = [hp_value["filter_num_deconv%d" % x] for x in range(1,hp_value["conv_num"])]
   
    return hp_value


def make_cnn(hp_value, nb_gpus):
    crop_shape = (hp_value["crop_shape"], hp_value["crop_shape"])
    # set our model
    img_dims, output_dims = crop_shape+(3,), crop_shape+(1,)
    model_single_gpu = seunet_model.seunet(img_dims, output_dims, filter_list_encoding=hp_value["filter_list_encoding"], filter_list_decoding=hp_value["filter_list_decodng"])
    print(nb_gpus)
    if int(nb_gpus) > 1:
        model_multi_gpu = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
    else:
        model_multi_gpu = model_single_gpu
    
    return model_single_gpu, model_multi_gpu


def random_search(iteration_num=1,
                  path_to_cnn_format = "./output/mm%02ddd%02d_%02d/", # % (now.month, now.day, count)
                  train_ids=np.arange(21,39),
                  data_size_per_epoch=2**14,
                  validation_ids=np.array([39,40]),
                  val_data_size = 2048,
                  epochs=256,
                  data_shape=(584,565),
#                  if_save_img=True,
                  nb_gpus=1,
                  patience = 8,
#                  epoch_num_fix = 0,
                  ):
    
    path_to_cnn_format = "../output/mm%02ddd%02d_%02d/"
    # make a folder to save history and models
    now = datetime.datetime.now()
    for count in range(100):
        path_to_cnn = path_to_cnn_format % (now.month, now.day, count)
        if not os.path.exists(path_to_cnn):
            os.mkdir(path_to_cnn)
            break
    
    # 使った python ファイルを cnn のフォルダに
    for file_py in os.listdir("./"):
        if re.match(".*.py$", file_py):
            path_to_code = "./"+file_py
            path_to_code_moved = path_to_cnn + file_py
            shutil.copyfile(path_to_code, path_to_code_moved)
        
    for iteration_id in range(iteration_num):
        
        try:
            # set path to save model and weight
            path_to_model_dir = path_to_cnn + "model%03d/" % iteration_id
            path_to_hp = path_to_model_dir + "hp%03d.csv" % iteration_id
#            path_to_save_model = path_to_model_dir + "model_epoch=%03d.h5"
            path_to_save_weights = path_to_model_dir + "weights_epoch=%03d.h5"
            # make directory
            os.makedirs(path_to_model_dir)
            
            hp_value = chose_hyperparam()
            crop_shape = (hp_value["crop_shape"], hp_value["crop_shape"])
            batch_size = hp_value["batch_size"]
            model_single_gpu, model_multi_gpu = make_cnn(hp_value,nb_gpus)
            steps_per_epoch=data_size_per_epoch//batch_size
            
        
            # load data
            train_images, train_manuals = load_image_manual(image_ids=train_ids,data_shape=data_shape)
        #    validation_images, validation_manuals = \
        #        load_image_manual(image_ids=validation_ids,data_shape=data_shape,crop_shape=crop_shape)
            val_data, val_label = make_validation_dataset(validation_ids=validation_ids,
                                                          load = True,
                                                          val_data_size = val_data_size,
                                                          data_shape=data_shape,
                                                          crop_shape=crop_shape,
                                                          )
                
            train_gen = batch_iter(images=train_images,
                                   manuals=train_manuals, 
                                   crop_shape=crop_shape,
                                   steps_per_epoch=steps_per_epoch,
                                   batch_size=batch_size,
                                   )
            if hp_value["loss"] == "binary_crossentropy":
                loss = 'binary_crossentropy'
            elif hp_value["loss"] == "mean_dice_coef_loss":
                loss = seunet_main.mean_dice_coef_loss
            opt_generator = Adam(lr=hp_value["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model_multi_gpu.compile(loss=loss, optimizer=opt_generator)
        
        
            validation_accuracy_ref, count_early_stopping = 0, 0
            epoch=0
            while epoch < epochs:
                epoch+=1
#            for epoch in range(1,epochs+1):
                model_multi_gpu.fit_generator(train_gen,
                                              steps_per_epoch=steps_per_epoch,
                                              epochs=1,
        #                                      epochs=epochs,
        #                                      callbacks=callbacks,
                                              validation_data=(val_data,val_label)
                                              )
                print('Epoch %s/%s done' % (epoch, epochs))
                print("")
                
                if epoch>0 and epoch % 1==0:
                    print(epoch)
#                    model_single_gpu.save(path_to_save_model % (epoch))
                    model_single_gpu.save_weights(path_to_save_weights % (epoch))
                    validation_accuracy = evaluation.whole_slide_accuracy(path_to_cnn=path_to_model_dir,
                                                                          epoch=epoch,
                                                                          path_to_model_weights=path_to_save_weights % (epoch),
                                                                          model=model_multi_gpu,
                                                                          image_ids=validation_ids,
                                                                          data_shape=data_shape,
                                                                          crop_shape=crop_shape,
                                                                          if_save_img=False,
                                                                          nb_gpus=nb_gpus,
                                                                          batch_size=batch_size,
                                                                          )
                    os.remove(path_to_save_weights % (epoch))
                    model_single_gpu.save_weights(path_to_save_weights[:-3] % (epoch) + "valacc=%.5f.h5" % validation_accuracy)                    
                    print("validation_accuracy = ", validation_accuracy)
                    
                    # setting early stopping
                    if validation_accuracy_ref < validation_accuracy:
                        validation_accuracy_ref = validation_accuracy
                        count_early_stopping = 0
                    elif validation_accuracy_ref >= validation_accuracy:
                        count_early_stopping += 1
                    print("count_early_stopping = ", count_early_stopping)
                    if count_early_stopping >= patience:
                        epoch=epochs
                                            
        except:
            print("skip %d" % iteration_id)

        # 空のフォルダは削除
#        if len(os.listdir(path_to_model_dir))==0:
#            os.rmdir(path_to_model_dir)
        # 精度の低いモデルの削除
        count_good_model = 0
#        if os.path.isdir(path_to_model_dir):
        for saved_model in os.listdir(path_to_model_dir):
            if re.match(".*.h5", saved_model):
                if float( saved_model[-10:-3] ) < 0.947:
                    os.remove(path_to_model_dir+saved_model)
                else:
                    count_good_model += 1
        # 空ディレクトリの削除
        if count_good_model==0:
            shutil.rmtree(path_to_model_dir)
#        # 空のフォルダは削除
#        if len(os.listdir(path_to_model_dir))==0:
#            os.rmdir(path_to_model_dir)
        # 最高精度のモデル以外を削除
        if os.path.isdir(path_to_model_dir):
            val_acc_max = 0
            for saved_model in os.listdir(path_to_model_dir):
                if os.path.isfile(path_to_model_dir+saved_model):
                    val_acc_model = int( re.search('(?<=valacc=0.)\d+', saved_model).group(0) )
                    if val_acc_model > val_acc_max:
                        val_acc_max = val_acc_model+0
            for saved_model in os.listdir(path_to_model_dir):
                if os.path.isfile(path_to_model_dir+saved_model):
                    val_acc_model = int( re.search('(?<=valacc=0.)\d+', saved_model).group(0) )
                    if val_acc_model < val_acc_max:
                        os.remove(path_to_model_dir+saved_model)
                    
        # save hyperparameters
        if os.path.isdir(path_to_model_dir):
            hp = open(path_to_hp, 'w')
            writer = csv.writer(hp, lineterminator='\n') 
            writer.writerow(["iteration_id",] + list(hp_value.keys()) + ["val_acc_max",]) # headder
            writer.writerow([iteration_id,] + list(hp_value.values()) + [val_acc_max,]) # headder
            hp.close()
            
        if os.path.isdir(path_to_model_dir):
            # モデルへのパスを指定
            path_to_model_weights = path_to_model_dir + os.listdir(path_to_model_dir)[0]
            # validation set に対する出力
            for image_id in validation_ids:
                evaluation.whole_slide_prediction(path_to_cnn=path_to_model_dir,
                                                  epoch=epoch,
                                                  path_to_model_weights=path_to_model_weights,
                                                  model=model_multi_gpu,
                                                  image_id=image_id,
                                                  data_shape=data_shape,
                                                  crop_shape=crop_shape,
                                                  nb_gpus=nb_gpus,
                                                  if_save_img=True,
                                                  if_save_npy=True,
                                                  batch_size=batch_size,
                                                  )
            # test set に対する出力
            for image_id in range(1,21):
                evaluation.whole_slide_prediction(path_to_cnn=path_to_model_dir,
                                                  epoch=epoch,
                                                  path_to_model_weights=path_to_model_weights,
                                                  model=model_multi_gpu,
                                                  image_id=image_id,
                                                  data_shape=data_shape,
                                                  crop_shape=crop_shape,
                                                  nb_gpus=nb_gpus,
                                                  dataset="test",
                                                  if_save_img=True,
                                                  if_save_npy=True,
                                                  batch_size=batch_size,
                                                  )
            # test set に対する精度を名前とするフォルダを作成
            evaluation.group_accuracies(group="validation",
                                        path_to_predict_dir=path_to_model_dir,
                                        image_ids=np.arange(39,41),
                                        thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999],
                                        )
            evaluation.group_accuracies(group="test",
                                        path_to_predict_dir=path_to_model_dir,
                                        image_ids=np.arange(1,21),
                                        thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999],
                                        )
#            testacc = evaluation.test_accuracy(path_to_gt_dir="../test/1st_manual/",
#                                               path_to_predict_dir=path_to_model_dir,
#                                               threshold=0.5,
#                                               )
#            os.mkdir(path_to_cnn + "testacc=%f/" % testacc)

def main():
#    filter_list_encoding = [64, 64, 128, 128, 256, 256, 512]
#    filter_list_decoding = [256, 128, 128, 64, 64, 64]
#    train(train_ids=np.arange(21,39),
#          validation_ids=np.arange(39,41),
#          val_data_size=2048,
#          batch_size=32,
#          data_size_per_epoch=2**14,
#          epochs=128, #256,
#          data_shape=(584,565),
#          crop_shape=(128,128),
#          filter_list_encoding=filter_list_encoding,
#          filter_list_decoding=filter_list_decoding,
#          if_save_img=True,
#          nb_gpus=1
#          )   
    image_id=1
    f = open("../segmentation_training_set/image%02d_mask.txt" % image_id)
    line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)
    shape = list(map(int, line.split()))
    print(shape)
    
    mask = np.zeros(shape[0]*shape[1])
    count = 0
    line = f.readline()
    while line:
#        if count==0:
#            print(line.split())
        count += 1
#        print(int(line))
        mask[count-1] = line
        line = f.readline()
    f.close
    mask = mask.reshape((shape[1],shape[0]))
    print(mask[0,0], np.amax(mask))
    mask[mask>0] = 255
    Image.fromarray(mask).show()
#    plt.imshow(mask)
#    random_search(iteration_num=100,
#                  path_to_cnn_format = "./cnn/mm%02ddd%02d_%02d/", # % (now.month, now.day, count)
#                  train_ids=np.arange(21,39),
#                  data_size_per_epoch=2**14,
#                  validation_ids=np.array([39,40]),
#                  val_data_size = 2048,
#                  epochs=256,
#                  data_shape=(584,565),
##                  if_save_img=True,
#                  nb_gpus=2,
#                  patience = 8,
##                  epoch_num_fix = 0,
#                  )    
    
if __name__ == '__main__':
    main()
    