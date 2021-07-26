import copy
import random
import sys
import numpy as np
from sklearn.metrics import f1_score, precision_score,recall_score
import glob

predicted_result_npy_path = ""
prediction_mask_npy_path = ""
predition_label_npy_path = ""

preds_test_mod = np.load(predicted_result_npy_path)
prediction_mask_npy = np.load(prediction_mask_npy_path)
predition_label_npy = np.load(predition_label_npy_path)

dim = predition_label_npy_path.shape
numr = dim[0]//(224 - buf*2)
numc = dim[1]//(224 - buf*2)
count = -1
for i in range(numr):
    
    if(i == 20):
        print("row: ",i,"column: ",j, "count", count)
        break 
        
    for j in range(int(numc/2)-1):
        count += 1    
        temp = preds_test_mod[count][buf:-buf,buf:-buf]
        if j == 0:
            rows = temp
        else:
            rows = np.concatenate((rows,temp),axis = 1)
            
    if i == 0:
        prediction_map = copy.copy(rows)
    else:
        prediction_map = np.concatenate((prediction_map,rows),axis = 0)

prediction_map = prediction_map[:,:,0]
print("prediction_map",prediction_map.shape)

# mask
mask = prediction_mask_npy[:prediction_map.shape[0],:prediction_map.shape[1]]
[lr,lc] = np.where(mask == 1)
print("mask",mask.shape)

# Read reference data
groundtruth = predition_label_npy[:prediction_map.shape[0],:prediction_map.shape[1]]
groundtruthlist = predition_label_npy[:prediction_map.shape[0],:prediction_map.shape[1]][lr,lc]
prediction = np.logical_and(prediction_map,mask)
predictionlist = np.logical_and(prediction_map,mask)[lr,lc]

print('F1 score of Nonstream: '+str(f1_score(groundtruthlist, predictionlist,labels=[0], average = 'micro')))
print('F1 score of Stream: '+str(f1_score(groundtruthlist, predictionlist,labels=[1], average = 'micro')))

print('Precision of Nonstream: '+str(precision_score(groundtruthlist, predictionlist,labels=[0], average = 'micro')))
print('Precision of Stream: '+str(precision_score(groundtruthlist, predictionlist,labels=[1], average = 'micro')))

print('Recall of Nonstream: '+str(recall_score(groundtruthlist, predictionlist,labels=[0], average = 'micro')))
print('Recall of Stream: '+str(recall_score(groundtruthlist, predictionlist,labels=[1], average = 'micro')))