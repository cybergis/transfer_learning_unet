from PIL import Image as im
im.MAX_IMAGE_PIXELS = None
import numpy as np
import glob
import sys

# total arguments
n = len(sys.argv)

# Update the organized data
# Order the files according to the name of the files
if(sys.argv[2]):
    data_folder = sys.argv[2]
else:
    data_folder = './organized_data/'


if(sys.argv[1] == "with_NAIP"):
    # Files' name of all the ionput data
    files = [
        '01_CovingtonRiver_020801030302_CURVATURE_nodata.tif',
        '04_CovingtonRiver_020801030302_DEM_nodata.tif', 
        '05_CovingtonRiver_020801030302_TPI_21_nodata.tif', 
        '07_CovingtonRiver_020801030302_Cov10cell_Geomorphon.tif',  
        #NAIP images are in range 0-255 already
        'NAIP/NAIP2018_RED_edit_nodata.tif',
        'NAIP/NAIP2018_GREEN_edit_nodata.tif',
        'NAIP/NAIP2018_BLUE_edit_nodata.tif',
        'NAIP/NAIP2018_INFARED_edit_nodata.tif' 
        ]

    #The ranges here are the range of 3SD or 99.7% of all data
    #This is to prevent problem in the normalization process
    ranges =[
        [-0.28222607489286866,0.28248185206077936], # '01_CovingtonRiver_020801030302_CURVATURE_nodata.tif'
        [-207.05215950623995, 938.84507252262], # '04_CovingtonRiver_020801030302_DEM_nodata.tif'
        [-0.8616536368618285, 0.8618900909336314], # '05_CovingtonRiver_020801030302_TPI_21_nodata.tif'
        [3.1338925783175906, 8.81583142566701], # '07_CovingtonRiver_020801030302_Cov10cell_Geomorphon.tif'
        ]
else:
    # Files' name of all the ionput data
    files = [
        '01_CovingtonRiver_020801030302_CURVATURE_nodata.tif',
        '02_CovingtonRiver_020801030302_SLOPE_nodata.tif', 
        '03_CovingtonRiver_020801030302_OPENESS_nodata.tif',
        '04_CovingtonRiver_020801030302_DEM_nodata.tif', 
        '05_CovingtonRiver_020801030302_TPI_21_nodata.tif', 
        '06_CovingtonRiver_020801030302_INTENSITY_nodata.tif',
        '07_CovingtonRiver_020801030302_Cov10cell_Geomorphon.tif', 
        '08_TPI_CovingtonRiver_020801030302_3_nodata.tif',
        ]

    #The ranges here are the range of 3SD or 99.7% of all data
    #This is to prevent problem in the normalization process
    ranges =[
        [-0.28222607489286866,0.28248185206077936], # '01_CovingtonRiver_020801030302_CURVATURE_nodata.tif'
        [-0.28097646432537005,0.7456170822778501], # '02_CovingtonRiver_020801030302_SLOPE_nodata.tif'
        [81.4260223726079, 96.2421904790021], # '03_CovingtonRiver_020801030302_OPENESS_nodata.tif'
        [-207.05215950623995, 938.84507252262], # '04_CovingtonRiver_020801030302_DEM_nodata.tif'
        [-0.8616536368618285, 0.8618900909336314], # '05_CovingtonRiver_020801030302_TPI_21_nodata.tif'
        [1.494045913614002, 77.145486958992],# '06_CovingtonRiver_020801030302_INTENSITY_nodata.tif'
        [3.1338925783175906, 8.81583142566701], # '07_CovingtonRiver_020801030302_Cov10cell_Geomorphon.tif'
        [-0.19840276259099499, 0.198406762590995] # '09_TPI_CovingtonRiver_020801030302_3_nodata.tif'
        ]



#initialize the output
output = [] 
output = np.array(output)

def normalize(array_x, min_x, max_x):

	array_x[ (array_x!=-9999) & (array_x < min_x) ] = min_x
	array_x[ (array_x!=-9999) & (array_x > max_x) ] = max_x

	print(min_x , max_x)

	array_x[array_x!=-9999] = ((array_x[array_x!=-9999]-min_x)/(max_x-min_x))*255
	output = array_x
	
	return output

for num, file in enumerate(files, start = 0 ):
		
	path = data_folder+file
	print(path)

	image = im.open(path)
	image_array = np.array(image)
	print(image_array.shape)
	
	if (num == 0):
		output=np.empty((len(files),image_array.shape[0], image_array.shape[1]))

	if(files == '07_CovingtonRiver_020801030302_Cov10cell_Geomorphon.tif'):
		image_array[image_array==255] = -9999
		image_array = normalize(image_array,ranges[num][0],ranges[num][1])

	elif("NAIP" in file):
		image_array[image_array==0] = -9999

	else:
		image_array = normalize(image_array,ranges[num][0],ranges[num][1])

	print("Normalized ", path)
	# print(image_array)		
	output[num] = image_array

output = np.moveaxis(output,0,-1)

np.save('total_without_NAIP',output)
print('saved total')

def generate_mask():
	
	path = './organized_data/01_CovingtonRiver_020801030302_CURVATURE_nodata.tif'

	image = im.open(path)
	image_array = np.array(image)
	
	# image_array = image_array[1:image_array.shape[0]-1,1:image_array.shape[1]-1]
	
	print(image_array.shape)

	mask = image_array != -9999
	mask.astype(int)
	np.save('mask',mask)

generate_mask()
print('save mask')

def generate_reference():
	path = './organized_data/reference_nodata.tif'

	image = im.open(path)
	image_array = np.array(image)	

	image_array[image_array==-9999] = np.nan
	
	np.save('reference_as_None',image_array)

generate_reference()
print('saved reference')
