## Load all the dependencies
import os
import sys
import random
import warnings
import numpy as np
from itertools import chain
from numpy import genfromtxt
from tensorflow import random
from keras import backend as K
# from keras import backend as k
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Layer, UpSampling2D, GlobalAveragePooling2D, Multiply, Dense, Reshape, Permute, multiply, dot, add, Input
from keras.layers.core import Dropout, Lambda, SpatialDropout2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model, model_from_yaml, Sequential
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

np.random.seed(1337) # for reproducibility
random.set_seed(1337)
print(tf.__version__)

name = "model_fine_tuning_"

# Use dice coefficient function as the loss function 
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

# Jacard coefficient
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

# calculate loss value
def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

# calculate loss value
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def Residual_CNN_block(x, size, dropout=0.0, batch_norm=True):
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
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    return conv

class multiplication(Layer):
    def __init__(self,inter_channel = None,**kwargs):
        super(multiplication, self).__init__(**kwargs)
        self.inter_channel = inter_channel
    def build(self,input_shape=None):
        self.k = self.add_weight(name='k',shape=(1,),initializer='zeros',dtype='float32',trainable=True)
    def get_config(self):
        base_config = super(multiplication, self).get_config()
        config = {'inter_channel':self.inter_channel}
        return dict(list(base_config.items()) + list(config.items()))  
    def call(self,inputs):
        g,x,x_query,phi_g,x_value = inputs[0],inputs[1],inputs[2],inputs[3],inputs[4]
        h,w,c = int(x.shape[1]),int(x.shape[2]),int(x.shape[3])
        x_query = K.reshape(x_query, shape=(-1,h*w, self.inter_channel//4))
        phi_g = K.reshape(phi_g,shape=(-1,h*w,self.inter_channel//4))
        x_value = K.reshape(x_value,shape=(-1,h*w,c))
        scale = dot([K.permute_dimensions(phi_g,(0,2,1)), x_query], axes=(1, 2))
        soft_scale = Activation('softmax')(scale)
        scaled_value = dot([K.permute_dimensions(soft_scale,(0,2,1)),K.permute_dimensions(x_value,(0,2,1))],axes=(1, 2))
        scaled_value = K.reshape(scaled_value, shape=(-1,h,w,c))        
        customize_multi = self.k * scaled_value
        layero = add([customize_multi,x])
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
        concate = my_concat([layero,g])
        return concate 
    def compute_output_shape(self,input_shape):
        ll = list(input_shape)[1]
        return (None,ll[1],ll[1],ll[3]*3)
    def get_custom_objects():
        return {'multiplication': multiplication}

def attention_up_and_concatenate(inputs):
    g,x = inputs[0],inputs[1]
    inter_channel = g.get_shape().as_list()[3]
    g = Conv2DTranspose(inter_channel, (2,2), strides=[2, 2],padding='same')(g)
    x_query = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(x)
    phi_g = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(g)
    x_value = Conv2D(inter_channel//2, [1, 1], strides=[1, 1], data_format='channels_last')(x)
    inputs = [g,x,x_query,phi_g,x_value]
    concate = multiplication(inter_channel)(inputs)
    return concate

class multiplication2(Layer):
    def __init__(self,inter_channel = None,**kwargs):
        super(multiplication2, self).__init__(**kwargs)
        self.inter_channel = inter_channel
    def build(self,input_shape=None):
        self.k = self.add_weight(name='k',shape=(1,),initializer='zeros',dtype='float32',trainable=True)
    def get_config(self):
        base_config = super(multiplication2, self).get_config()
        config = {'inter_channel':self.inter_channel}
        return dict(list(base_config.items()) + list(config.items()))  
    def call(self,inputs):
        g,x,rate = inputs[0],inputs[1],inputs[2]
        scaled_value = multiply([x, rate])
        att_x =  self.k * scaled_value
        att_x = add([att_x,x])
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
        concate = my_concat([att_x, g])
        return concate 
    def compute_output_shape(self,input_shape):
        ll = list(input_shape)[1]
        return (None,ll[1],ll[1],ll[3]*2)
    def get_custom_objects():
        return {'multiplication2': multiplication2}

def attention_up_and_concatenate2(inputs):
    g, x = inputs[0],inputs[1]
    inter_channel = g.get_shape().as_list()[3]
    g = Conv2DTranspose(inter_channel//2, (3,3), strides=[2, 2],padding='same')(g)
    g = Conv2D(inter_channel//2, [1, 1], strides=[1, 1], data_format='channels_last')(g)
    theta_x = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(x)
    phi_g = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format='channels_last')(f)
    rate = Activation('sigmoid')(psi_f)
    concate =  multiplication2()([g,x,rate])
    return concate

loaded_model = load_model('original_model/model_augv_attention2.h5',custom_objects={'multiplication': multiplication,'multiplication2': multiplication2,'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})

training_layers = 'First4_'

# remove the last 2 layer using pop() function
loaded_model.layers.pop()
loaded_model.layers.pop()

for (index, layer) in enumerate(loaded_model.layers):
    if (index < 4):
        layer.trainable = True
    else:
        layer.trainable = False

# Create new model from the model using the input and output of the last layer (after poping last 2 layers)
model_without_last = Model(loaded_model.input,  loaded_model.layers[-1].output)

# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1

# 1 dimensional convolution and generate probabilities from Sigmoid function
conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1), name='conv2d_last')(model_without_last.output)
new_out = Activation('sigmoid', name='activation_last')(conv_final)

# Created new model with the newly added last two layers 
transfered_model = Model(inputs=model_without_last.input, outputs=new_out)


data_path = 'samples/experiment_2/'
# read in training and validation data
X_train = np.load(data_path+'train_data.npy')
Y_train = np.load(data_path+'train_label.npy')
X_Validation = np.load(data_path+'vali_data.npy')
Y_Validation = np.load(data_path+'vali_label.npy')
print(X_train.shape)
print(X_Validation.shape)

# Define the input patch size which we use 224 pixels by 224 pixels
patch_size = 224
IMG_WIDTH = patch_size
IMG_HEIGHT = patch_size

# Number of feature channels or raster images 
INPUT_CHANNELS = 8

# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1

# train for maximum 25 epochs
maxepoch = 25

# hyperparameters
learning_rate = 0.0000359
patience = 20
transfered_model.compile(optimizer=Adam(lr=learning_rate),loss = dice_coef_loss, metrics=[dice_coef,'accuracy'])
callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=patience, min_lr=1e-9, verbose=1, mode='min'),
        EarlyStopping(monitor='val_loss', patience=patience+10, verbose=0),
        ModelCheckpoint('first_pass_tf_model.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

fine_tuned_model_P1_history = transfered_model.fit(X_train_new, Y_train, validation_data=(X_Validation_new,Y_Validation), batch_size=2, epochs=maxepoch, callbacks=callbacks)

for (index, layer) in enumerate(transfered_model.layers):
    layer.trainable = True


# Define the input patch size which we use 224 pixels by 224 pixels
patch_size = 224
IMG_WIDTH = patch_size
IMG_HEIGHT = patch_size

# Number of feature channels or raster images 
INPUT_CHANNELS = 8

# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1

# train for maximum 25 epochs
maxepoch = 25

# hyperparameters
# 10 times smaller learning rate
learning_rate = 0.000003359
patience = 25
transfered_model.compile(optimizer=Adam(lr=learning_rate),loss = dice_coef_loss, metrics=[dice_coef,'accuracy'])
callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=patience, min_lr=1e-9, verbose=1, mode='min'),
        EarlyStopping(monitor='val_loss', patience=patience+10, verbose=0),
        ModelCheckpoint('second_pass_tf_model.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

fine_tuned_model_P2_history = transfered_model.fit(X_train_new, Y_train, validation_data=(X_Validation_new,Y_Validation), batch_size=1, epochs=maxepoch, callbacks=callbacks)

import pickle
name = "experiment_2_"+training_layers+"model"

# save the trained model
root_path = './training_results/experiment_2/'
    
# save the weights as h5 file
transfered_model.save(root_path+name+".h5")

# save the intermdediate results and training statistics
with open(root_path+name+".pickle", 'wb') as file_pi:
    pickle.dump(fine_tuned_model_P2_history.history, file_pi, protocol=2)