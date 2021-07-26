import os
import numpy as np
import sys
import copy
import random
from datetime import datetime

# total arguments
n = len(sys.argv)
if(n < 5):
	raise Exception("The command must have 4 parameters: \n 1. path to total data \n 2. path to mask file \n 3. path to reference \n 4. path to output folder")
else:
	totaldata_path = sys.argv[1]
	mask_path = sys.argv[2]
	label_path = sys.argv[3]
	output_path = sys.argv[4]

# stream/non-stream sample size
patch_size = 224 #patch size of each sample

#Total data dimension: 13927, 14466
totaldata = np.load(totaldata_path)
mask = np.load(mask_path)
label = np.load(label_path)

#Add mask 
totaldata = np.concatenate((totaldata,mask[:,:,np.newaxis]),axis = 2)

print(totaldata.shape)
print('Completed: Data Loading!')

# buffer size
buf = 30

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
np.save(output_path+"/test_data.npy",total)
print("Test data is generated!")
