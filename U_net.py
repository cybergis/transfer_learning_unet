import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
from numpy import genfromtxt
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt



np.random.seed(1337) # for reproducibility

from tensorflow import set_random_seed
set_random_seed(1337)
#from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.layers import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

#####
it = 'full'
# Set some parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224
# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 5
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1
#seed = 42
#random.seed = seed
#np.random.seed = seed

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def double_conv_layer(x, size, dropout=0.0, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    dice_list = [s for s in history.history.keys() if 'dice' in s and 'val' not in s]
    val_dice_list = [s for s in history.history.keys() if 'dice' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Dice Coefficient
    plt.figure(2)
    for l in dice_list:
        plt.plot(epochs, history.history[l], 'b', label='Training Dice Coefficient (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_dice_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation Dice Coefficient (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    
    ## Accuracy
    plt.figure(3)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
################
def UNET_224(dropout_val=0.2, weights=None):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, IMG_WIDTH, IMG_WIDTH))
        axis = 1
    else:
        inputs = Input((IMG_WIDTH, IMG_WIDTH, INPUT_CHANNELS))
        axis = 3
    filters = 32

    conv_224 = double_conv_layer(inputs, filters)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16*filters)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="UNET_224")

    #if weights == 'generator' and axis == 3 and INPUT_CHANNELS == 3 and OUTPUT_MASK_CHANNELS == 1:
    ##    weights_path = get_file(
     #       'unet_224_weights_tf_dim_ordering_tf_generator.h5',
     #       UNET_224_WEIGHT_PATH,
     #       cache_subdir='models',
     #       file_hash='203146f209baf34ac0d793e1691f1ab7')
     #   model.load_weights(weights_path)

    return model

import os
os.chdir('/home/cc/move/')
X_train = np.load('train_data_PC'+it+'.npy')#[:500]#[:2000]
Y_train = np.load('train_label_PC'+it+'.npy')#[:500]#[:,:,:,np.newaxis]#[:2000]
X_Validation = np.load('vali_data_PC'+it+'.npy')#[:500]#[:700]
Y_Validation = np.load('vali_label_PC'+it+'.npy')#[:500]#[:,:,:,np.newaxis]#[:700]

def delete5(array):
    return np.delete(array,4,axis = 3)
def deletespe(array):
    return array[:,:,:,:4]
#X_train = delete5(X_train)
#X_Validation = delete5(X_Validation)

# Only curvature,tpi1, tpi2, geomorphon
#X_train = deletespe(X_train)
#X_Validation = deletespe(X_Validation)



#import pdb
#pdb.set_trace()

model = UNET_224()
learning_rate =0.0001
patience = 20
model.compile(optimizer=Adam(lr=learning_rate),loss = dice_coef_loss,metrics=[dice_coef,'accuracy'])
callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('unetLr03Full.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]
# Fit model
#results_03 = model.fit(X_train, Y_train, validation_split=0.4, batch_size=16, epochs=50,
#                    callbacks=callbacks)
#X_train = X_train[:20]
#Y_train = Y_train[:20]
#X_Validation = X_Validation
#Y_Validation = Y_Validation
results_03 = model.fit(X_train, Y_train, validation_data=(X_Validation,Y_Validation), batch_size=32, epochs=300,
                   callbacks=callbacks)

X_test = np.load('/home/cc/move/prediction_data_PC'+it+'.npy')
preds_test = model.predict(X_test)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
#Y_test =np.load("/data/cigi/scratch/zeweixu2/USGS_keras/True_label_PC.npy")

#mappedResult = []
#for i in range(0,X_test.shape[0]):
#    curGT = Y_test[i]
#    curPD = preds_test_t[i]
#    values = curGT.copy()
#    for x in range(0,IMG_WIDTH):
#        for y in range(0,IMG_WIDTH):
#            if curGT[x,y]==0 and curPD[x,y]==0:
#                #print("here")
#                values[x,y] = 0
#            elif curGT[x,y]==0 and curPD[x,y]==1:
#                #print("here1")
#                values[x,y] = 1
#            elif curGT[x,y]==1 and curPD[x,y]==0:
#                #print("here2")
#                values[x,y] = 2
#            elif curGT[x,y]==1 and curPD[x,y]==1:
#                values[x,y] = 3
#    mappedResult.append(values)
#mappedResult = np.asarray(mappedResult)

from keras.models import load_model
model.save('total_PC'+it+'.h5')
import pickle
with open('SL_total_PC'+it+'.pickle', 'wb') as file_pi:
        pickle.dump(results_03.history, file_pi, protocol=2)   

#import pickle
#with open('SL_1500_spectral.pickle', 'wb') as file_pi:
#        pickle.dump(results_03.history, file_pi)

#np.save('SL_total.npy',results_03.history)
#np.save('TestmappedResultotal_PCfull.npy',mappedResult)
np.save('preds_test_total_PC'+it+'.npy',preds_test_t)

