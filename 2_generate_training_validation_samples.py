####### Generate training/validation/testing image patches 
import copy
import random
import sys
import numpy as np
from intervaltree import Interval, IntervalTree

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
# size = 2000 # number of samples 
patch_size = 224 #patch size of each sample

#Total data dimension: 14406*13867
totaldata = np.load(totaldata_path)
mask = np.load(mask_path)

totaldata = np.concatenate((totaldata,mask[:,:,np.newaxis]),axis = 2)
label = np.load(label_path)

print('Completed: Data Loading!')

# Use the upper half to generate training and validation data
half = totaldata.shape[0]//2
label_train_vali = label[:half]
[r,c] = np.where(label_train_vali == True) #steamline patches
[rn,cn] = np.where(label_train_vali == False) #non-steamline patches

print(len(r),len(rn))

inder = random.sample(range(0, len(r)-1), 100000)
indenr = random.sample(range(0, len(rn)-1), 100000)
r,c = r[inder],c[inder]
rn,cn = rn[indenr],cn[indenr]

print('Get all data coordinates!')

samples_row = IntervalTree()
samples_column = IntervalTree()
validation_sample = []
training_sample = []

start_index = 0

def is_invalid(mode, temp, templ, minrow, maxrow, mincolumn, maxcolumn):

	# check complete patch
	is_not_complete_patch = np.any(temp[0,:,:,-1] <= 0) or temp.shape[1:3] != (patch_size,patch_size) or templ.shape[1:] != (patch_size,patch_size)

	if mode =="training":
		# training sample can overlap with itself 
		# so, we only check if it is complete patch or not
		return is_not_complete_patch

	if mode =="validation":
		# validation sample **cannot** overlap with training samples patches
		# so, we need to 1) check the completeness 2) check if it overleps with training samples

		is_overlap = False

		row_overlap = samples_row[minrow:maxrow]
		column_overlap = samples_column[mincolumn:maxcolumn]

		for row_interval in row_overlap:
			begin, end, row_data = row_interval
			for column_interval in column_overlap:
				begin, end, column_data = column_interval
				if(row_data == column_data):
					is_overlap = True
					# print(minrow, maxrow, mincolumn, maxcolumn)
					# print(row_interval, column_interval)
					break
			else:
				continue
			break

		return bool(is_not_complete_patch or is_overlap)

#extract stream/non-stream samples using the random patches
def generate_samples(totaldata,row,col,label,size,train_or_vali):
	
	global start_index
	global samples_row
	global samples_column
	global training_sample
	global validation_sample

	count = 0
	row = row[start_index:]
	col = col[start_index:]

	for index,(i,j) in enumerate(zip(row,col)):

		# calculate min max ranges of the patch.
		minrow = (i-int(patch_size/2))
		maxrow = (i+int(patch_size/2))
		mincolumn = (j-int(patch_size/2))
		maxcolumn = (j+int(patch_size/2))

		# extract data from total dataset and label of total dataset
		temp = totaldata[minrow:maxrow,mincolumn:maxcolumn,:][np.newaxis,:,:,:]
		templ = label[minrow:maxrow,mincolumn:maxcolumn][np.newaxis,:,:]

		# validate the conditions
		if is_invalid(train_or_vali,temp,templ,minrow,maxrow,mincolumn,maxcolumn):
			# if not complete (or overlap in validation) skip this patch.
			continue 
		else:
			# if valid add to samples
			count += 1
			if count == 1:            
				train_vali = temp[:,:,:,:-1]
				train_vali_label = templ
			else:
				train_vali = np.concatenate((train_vali, temp[:,:,:,:-1]),axis = 0)
				train_vali_label = np.concatenate((train_vali_label, templ),axis = 0)
			
			if train_or_vali == "training":
				training_sample.append((minrow,mincolumn))

				# if it is training sample, add the sample patch x and y ranges to the trees with the count as the ranges' label.
				samples_row[minrow:maxrow] = count
				samples_column[mincolumn:maxcolumn] = count

			if train_or_vali == "validation":
				validation_sample.append((minrow,mincolumn))

		if count == size:
			print("Generated "+str(train_or_vali)+": "+str(count)+" samples and used from "+str(start_index)+" to "+str(start_index+index)+"random rows and columns")
			start_index = start_index+index
			return [train_vali,train_vali_label]
	
	print("Not enough random smaples")
	print("Generated "+str(train_or_vali)+": "+str(count)+" samples out of "+ str(size)+ " end at index "+str(index))
	return

print('Extracting data patches!')



[train_vali_stream,train_vali_stream_label] = generate_samples(totaldata,r,c,label_train_vali,300,"training")
[train_vali_nonstream,train_vali_nonstream_label] = generate_samples(totaldata,rn,cn,label_train_vali,300,"training")

trainvali_data = np.concatenate((train_vali_stream,train_vali_nonstream),axis = 0)
trainvali_label = np.concatenate((train_vali_stream_label,train_vali_nonstream_label),axis = 0)


# Shuffle training and validation samples
s = np.arange(trainvali_data.shape[0])
np.random.shuffle(s)
train_data = trainvali_data[s]
train_label = trainvali_label[s]

#Save the trainging samples both data and label
np.save(output_path+'/train_data.npy',train_data)
np.save(output_path+'/train_label.npy',train_label[:,:,:,np.newaxis])
training_sample = np.array(training_sample)
np.save(output_path+'/train_patches_top-left.npy',training_sample)


[train_vali_stream,train_vali_stream_label] = generate_samples(totaldata,r,c,label_train_vali,300,"validation")
[train_vali_nonstream,train_vali_nonstream_label] = generate_samples(totaldata,rn,cn,label_train_vali,300,"validation")

trainvali_data = np.concatenate((train_vali_stream,train_vali_nonstream),axis = 0)
trainvali_label = np.concatenate((train_vali_stream_label,train_vali_nonstream_label),axis = 0)

# Shuffle training and validation samples
s = np.arange(trainvali_data.shape[0])
np.random.shuffle(s)
vali_data = trainvali_data[s]
vali_label = trainvali_label[s]

#Save the validation samples both data and label
np.save(output_path+'/vali_data.npy',vali_data)
np.save(output_path+'/vali_label.npy',vali_label[:,:,:,np.newaxis])
validation_sample = np.array(validation_sample)
np.save(output_path+'/vali_patches_top-left.npy',validation_sample)