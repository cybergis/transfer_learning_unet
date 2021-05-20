import os
import numpy as np
import copy
import random
from datetime import datetime

# stream/non-stream sample size
patch_size = 224 #patch size of each sample

#Total data dimension: 13927, 14466
totaldata = np.load('./Total_data/total_without_NAIP.npy')
mask = np.load('./Total_data/mask.npy')
#Add mask 
totaldata = np.concatenate((totaldata,mask[:,:,np.newaxis]),axis = 2)

label = np.load('./Total_data/reference_nodata_as_0.npy')

print(totaldata.shape)
print(mask.shape)
print(label.shape)
print('Completed: Data Loading!')

# buffer size
buf = 30
it = 'full'
# Image dimension
IMG_WIDTH = 224
IMG_HEIGHT = 224

# moving window size = image_dimension - 2*buffer_size
mw = IMG_WIDTH - buf*2

half = totaldata.shape[0]//2
bottom_half_total = totaldata[half:,:,:]

bottom_half_mask = mask[half:,:]
print("mask",bottom_half_mask.shape)
np.save("./train_test_dataset/bottom_half_test_mask.npy",bottom_half_mask)
print('Saved Mask!')


bottom_half_label = label[half:,:]
print("label",bottom_half_label.shape)
np.save("./train_test_dataset/bottom_half_test_label.npy",bottom_half_label)
print('Saved Label!')

# Number of trainig channels
# Adding padding to width and height for moving window 
totalnew = np.pad(bottom_half_total, ((buf, buf),(buf,buf),(0,0)), 'symmetric')

#The last dimension is the mask 
#totalnew = totalnew[:,:,buf:(buf+9)]
totalnew = totalnew[:,:,(0,1,2,3,4,5,6,7)]
print(totalnew.shape)

#get taotal data height and width
dim = bottom_half_total.shape[:2]

# number of patch rows
numr = dim[0]//(IMG_WIDTH - buf*2)#224
print('rows:'+str(numr))

# number of patch columns
numc = dim[1]//(IMG_WIDTH - buf*2)#224
# only left side
# numc = dim[1]//2//(IMG_WIDTH - buf*2)#224
print('columns:'+str(numc))

# Splitting the total data into patches 4
count = 0
for i in range(numr):
    print("row: ",i)
    for j in range(numr):
        # print("column: ",j)
        count += 1
        temp = totalnew[i*mw:(i*mw+224),j*mw:(j*mw+224),:][np.newaxis,:,:,:]
        if count == 1:
            total = temp
        else:
            total = np.concatenate((total, temp),axis = 0)
        
print(total.shape)
# Save the total dataset
np.save("./train_test_dataset/bottom_half_test_data.npy",total)
print("Testing moving window is generate!")
