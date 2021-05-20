# from PIL import Image as im
# im.MAX_IMAGE_PIXELS = 933120000
import numpy as np
# import glob

# # Update the organized data
# # Order the files according to the name of the files
# data_folder = './organized_data/'

# files = [
# 	'01_CovingtonRiver_020801030302_CURVATURE_nodata.tif',
# 	'02_CovingtonRiver_020801030302_SLOPE_nodata.tif', 
# 	'03_CovingtonRiver_020801030302_OPENESS_nodata.tif',
# 	'04_CovingtonRiver_020801030302_DEM_nodata.tif', 
# 	'05_CovingtonRiver_020801030302_TPI_21_nodata.tif', 
# 	'06_CovingtonRiver_020801030302_INTENSITY_nodata.tif', 
# 	'07_CovingtonRiver_020801030302_Cov10cell_Geomorphon.tif', 
# 	'08_TPI_CovingtonRiver_020801030302_9_nodata.tif', 
# 	'09_TPI_CovingtonRiver_020801030302_3_nodata.tif', 
# 	'NAIP/NAIP2018_RED_edit_nodata.tif',
# 	'NAIP/NAIP2018_GREEN_edit_nodata.tif',
# 	'NAIP/NAIP2018_BLUE_edit_nodata.tif',
# 	'NAIP/NAIP2018_INFARED_edit_nodata.tif'
#     ]


# for num, file in enumerate(files, start = 0 ):
#     path = data_folder+file
#     print(path)
#     image = im.open(path)
#     image_array = np.array(image)
#     print(image_array[0,0])
#     print(image_array.shape)

#     # image_array[np.isnan(image_array)] = -99
#     # image_array[image_array <-99] = -99

#     # print(len(image_array[np.where( image_array > 1000 )]))

#     # min_x = np.min(image_array[image_array!=-99])
#     # max_x = np.max(image_array[image_array!=-99])

#     # print(min_x , max_x)
buf = 30
IMG_WIDTH = 224

# totaldata = np.load('./Total_data/total.npy')
# print(totaldata.shape)
# dim = totaldata.shape[:2]
# numr = dim[0]//(IMG_WIDTH - buf*2)
# print(numr)
# numc = dim[1]//(IMG_WIDTH - buf*2)
# print(numc)

# # totaldata = np.load('./Total_data/mask.npy')
# # print(totaldata.shape)
print('./Total_data/reference_nodata_as_0.npy')
totaldata = np.load('./Total_data/reference_nodata_as_0.npy')
print(totaldata.shape)
dim = totaldata.shape[:2]
numr = dim[0]//(IMG_WIDTH - buf*2)
print(numr)
numc = dim[1]//(IMG_WIDTH - buf*2)
print(numc)
print()

print('./Total_data/reference.npy')
totaldata = np.load('./Total_data/reference.npy')
print(totaldata.shape)
dim = totaldata.shape[:2]
numr = dim[0]//(IMG_WIDTH - buf*2)
print(numr)
numc = dim[1]//(IMG_WIDTH - buf*2)
print(numc)

print('./train_test_dataset/prediction_data_22082020.npy')
totaldata = np.load('./train_test_dataset/prediction_data_22082020.npy')
print(totaldata.shape)
dim = totaldata.shape[:2]
numr = dim[0]//(IMG_WIDTH - buf*2)
print(numr)
numc = dim[1]//(IMG_WIDTH - buf*2)
print(numc)

print('./train_test_dataset/prediction_data_nodata_as_02020-10-12_06-07-20_PM.npy')
totaldata = np.load('./train_test_dataset/prediction_data_nodata_as_02020-10-12_06-07-20_PM.npy')
print(totaldata.shape)
dim = totaldata.shape[:2]
numr = dim[0]//(IMG_WIDTH - buf*2)
print(numr)
numc = dim[1]//(IMG_WIDTH - buf*2)
print(numc)