from keras.utils import generic_utils
from keras.optimizers import Adam
import keras.backend as K
from keras.utils.training_utils import multi_gpu_model
import numpy as np
import os

from seunet_model import seunet

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection +1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def dice_coef2(y_true, y_pred):
    return 0.5*dice_coef(y_true, y_pred) + 0.5*dice_coef(1-y_true, 1-y_pred)


def mean_dice_coef_loss(y_true, y_pred):

    channels_nb = K.int_shape(y_pred)[-1]

    loss = 0

    for ch in range(channels_nb):
        loss = loss + dice_coef2(y_true[:,:,:,ch], y_pred[:,:,:,ch])

    return 1 - loss/channels_nb


def train(path_to_image, path_to_target, model_path, batch_size, nb_epoch, nb_gpus):

    # X_sketchが元の病理画像。[0,1]に規格化されたnp array
    X_sketch_train = np.load(path_to_image)
    # X_fullがsegmentationされた画像。[0,1]に規格化された4channel np array
    X_full_train = np.load(path_to_target)


    img_dim = X_full_train.shape[-4:]
    img_dim0 = X_sketch_train.shape[-4:]
    train_size = X_full_train.shape[0]

    if img_dim[:-1] != img_dim0[:-1]:
        print("Error: output shape must be the same as that of input")

    opt_generator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    generator_model = seunet(img_dim0, img_dim)
    
    print(nb_gpus)
    if int(nb_gpus) > 1:
        unet = multi_gpu_model(generator_model, gpus=nb_gpus)
    else:
        unet = generator_model

    unet.compile(loss=mean_dice_coef_loss, optimizer=opt_generator)

    print("Start training")
    for e in range(nb_epoch):

        e_shift = e+1

        unet.fit(X_sketch_train, X_full_train, batch_size=batch_size, epochs=1, validation_split=0.1)

        print('Epoch %s/%s done' % (e_shift, nb_epoch))
        print("")

#        model_path = pix2pix_path + '/models'

        if e % 10 == 9:

            if not os.path.exists(model_path):
                os.mkdir(model_path)

            gen_weights_path = model_path + '/seunet_weights_%s.h5' % (1+e)
            generator_model.save_weights(gen_weights_path, overwrite=True)


if __name__ == "__main__":

    import sys

    argvs = sys.argv

    path_to_image = argvs[1]
    path_to_target = argvs[2]
    model_path = argvs[3]
    batch_size = int(argvs[4])
    nb_epoch = int(argvs[5])
    nb_gpus = int(argvs[6])

    train(path_to_image, path_to_target, model_path, batch_size, nb_epoch, nb_gpus)
