# Load all the dependencies
import os
import sys
import random
import warnings
import numpy as np
from itertools import chain
from numpy import genfromtxt
from tensorflow import random
from keras import backend as K
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

np.random.seed(1337) # for reproducibility
random.set_seed(1337)



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

print('Satrt laoding the model')
# 
# model = load_model('./models/model_attention2_transfered_NAIP_08122020.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})

# Transfered trained with NAIP included 
# model = load_model('./models/model_transfered_NAIP_20201005-143648.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})

# TRansfered trained with with NAIP included added spatialdropout2D with 0.5 drop rate
# model = load_model('./models/model_transfered_NAIP_spatialdropout2D_0.5_20201005-113304.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})


######################################## Original Model ####################################################

# # # Original Model
# model = load_model('./models/model_augv_attention2.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})

############################################ With NAIP #########################################################
######################################## Transfer Learning  ####################################################
# # Transfer learning with NAIP 100 samples 
# model = load_model('./models/model_transfere-_learning_NAIP_100_samples_20201012-150904.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})

# # Transfer Learning with NAIP 350 samples 
# model = load_model('./models/model_transfere-_learning_NAIP_350_samples_20201012-155935.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})

# # Transfer Learning with NAIP 1000 samples 
# model = load_model('./models/model_transfere-_learning_NAIP_500_samples_20201012-170026.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})

######################################## Train from scratch ####################################################
# # Train from scratch with NAIP 100 samples 
# model = load_model('./models/model_train_from_scratch_NAIP_100_samples_20201012-174105.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})

# Train from scratch with NAIP 350 samples 
# model = load_model('./models/model_train_from_scratch_NAIP_350_samples_20201013-131534.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})

# # Train from scratch with NAIP 1000 samples 
# model = load_model('./models/model_train_from_scratch_NAIP_500_samples_20201012-204040.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})

############################################ Without NAIP #########################################################
############################################ Fine-Tuning ##########################################################

# # Fine-Tuning without NAIP 50 samples 
# model = load_model('./models/model_fine_tuning_No_NAIP_50_samples_50_epochs_20210111-142636.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_fine_tuning_No_NAIP_50_samples_50_epochs_20210111-142636.npy"

# # Fine-Tuning without NAIP 100 samples 
# model = load_model('./models/model_fine_tuning_learning_No_NAIP_100_samples20201116-102247.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_fine_tuning_learning_No_NAIP_100_samples20201116-102247.npy"

# # Fine-Tuning without NAIP 700 samples 
# model = load_model('./models/model_fine_tuning_learning_No_NAIP_700_samples20201116-125815.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_fine_tuning_learning_No_NAIP_700_samples20201116-125815.npy"

# # Fine-Tuning without NAIP 1200 samples 
# model = load_model('./models/model_fine_tuning_learning_No_NAIP_1200_samples20201116-024125.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_fine_tuning_learning_No_NAIP_1200_samples20201116-024125.npy"

############################################ Without Fine-Tuning ##########################################################                           

# # Transfer Learning without NAIP 60 samples 
# model = load_model('./models/model_transfere_learning_without_NAIP_60_samples_50_epochs_20201221-161116.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_transfere_learning_without_NAIP_60_samples_50_epochs_20201221-161116.npy"

# # # Transfer Learning without NAIP 50 samples 
# model = load_model('./models/model_transfere_learning_without_NAIP_50_samples_50_epochs_20201221-161359.h5', 
#                   custom_objects={'multiplication': multiplication,'multiplication2': multiplication2,
#                                   'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_transfere_learning_without_NAIP_50_samples_50_epochs_20201221-161359.npy"


# # Transfer Learning without NAIP 100 samples 
# model = load_model('./models/model_transfere_learning_without_NAIP_100_samples_200_epochs_20201123-101308.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_transfere_learning_without_NAIP_100_samples_200_epochs_20201123-101308.npy"

# # Transfer Learning without NAIP 700 samples 
# model = load_model('./models/model_transfere_learning_without_NAIP_700_samples_200_epochs_20201123-104330.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_transfere_learning_without_NAIP_700_samples_200_epochs_20201123-104330.npy"

# # Transfer Learning without NAIP 1200 samples 
# model = load_model('./models/model_transfere_learning_without_NAIP_1200_samples_200_epochs_20201123-114352.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_transfere_learning_without_NAIP_1200_samples_200_epochs_20201123-114352.npy"


############################################ Training from scratch ##########################################################   

# Training from scratch without NAIP 50 samples 
model = load_model('./models/model_train_from_scratch_No_NAIP_50_samples_50_epoch_20210111-140213.h5', 
custom_objects={'multiplication': multiplication,'multiplication2': multiplication2,
'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
results_path = "./results/prediction_result_train_from_scratch_No_NAIP_50_samples_50_epoch_20210111-140213.npy"

# # Training from scratch without NAIP 100 samples 
# model = load_model('./models/model_train_from_scratch_No_NAIP_100_samples_200_epoch_20201130-155732.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_train_from_scratch_No_NAIP_100_samples_200_epoch_20201130-155732.npy"

# # Training from scratch without NAIP 700 samples 
# model = load_model('./models/model_train_from_scratch_No_NAIP_700_samples_200_epoch_20201123-130148.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_train_from_scratch_No_NAIP_700_samples_200_epoch_20201123-130148.npy"

# # Training from scratch without NAIP 1200 samples 
# model = load_model('./models/model_train_from_scratch_No_NAIP_1200_samples_200_epoch_20201123-155557.h5', 
#                              custom_objects={'multiplication': multiplication,'multiplication2': multiplication2, 
#                                              'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef,})
# results_path = "./results/prediction_result_train_from_scratch_No_NAIP_1200_samples_200_epoch_20201123-155557.npy"




print('model loaded')
#Load test data
# #with_NAIP
# X_test = np.load('./train_test_dataset/prediction_data_nodata_as_02020-10-12_06-07-20_PM.npy')
#without_NAIP
X_test = np.load('./train_test_dataset/without_NAIP/prediction_data_2020-11-16_09-11-15_PM.npy')
print('load Data')
#predict the results
preds_test = model.predict(X_test)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
print('Start saving')
#Save prediction results
np.save(results_path,preds_test_t)
print('Finished saving')


