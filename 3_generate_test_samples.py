import os
import numpy as np
import copy
import random
from datetime import datetime

# stream/non-stream sample size
patch_size = 224 #patch size of each sample

#Total data dimension: 13927, 14466
totaldata = np.load('./Total_data/total.npy')
mask = np.load('./Total_data/mask.npy')
#Add mask 
totaldata = np.concatenate((totaldata,mask[:,:,np.newaxis]),axis = 2)

label = np.load('./Total_data/reference_nodata_as_0.npy')

print(totaldata.shape)
print('Completed: Data Loading!')

# buffer size
buf = 30
it = 'full'
# Image dimension
IMG_WIDTH = 224
IMG_HEIGHT = 224

# moving window size = image_dimension - 2*buffer_size
mw = IMG_WIDTH - buf*2


# Number of trainig channels
# Adding padding to width and height for moving window 
totalnew = np.pad(totaldata, ((buf, buf),(buf,buf),(0,0)), 'symmetric')
print(totalnew.shape)

#The last dimension is the mask 
#totalnew = totalnew[:,:,buf:(buf+9)]
totalnew = totalnew[:,:,(0,1,2,3,4,5,6,7)]
print(totalnew.shape)

#get taotal data height and width
dim = totaldata.shape[:2]

# number of patch rows
numr = dim[0]//(IMG_WIDTH - buf*2)#224
print('rows:'+str(numr))

# number of patch columns
numc = dim[1]//(IMG_WIDTH - buf*2)#224
# only left side
# numc = dim[1]//2//(IMG_WIDTH - buf*2)#224
print('columns:'+str(numc))

# Splitting the total data into patches
count = 0
for i in range(numr):
  print("row: ",i)
  for j in range(numr):
    # print("column: ",j)
    count += 1
    temp = totalnew[i*mw:(i*mw+224),j*mw:(j*mw+224),:][np.newaxis,:,:,:]
    if count == 1:
        total = temp#[:,:,:,:-1]
        #print(total.shape)
    else:
        total = np.concatenate((total, temp),axis = 0)
        
print(total.shape)
# Save the total dataset
folder_time = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
np.save("./train_test_dataset/prediction_data_nodata_as_0"+folder_time+".npy",total)
print("Testing moving window is generate!")
