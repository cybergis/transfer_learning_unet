# Generate prediction map
it = 'full' # full
buf = 30
# uppper left cordinates
# The top-left corner of Covinrton area
uc = [1443556.82396,-7784.43629754] # smallest x, largest y
# patch dimension
IMG_WIDTH = 224
from osgeo import gdal
def array_to_raster(array,xmin,ymax,row,col,proj,name):
    dst_filename = name
    x_pixels = col
    y_pixels = row
    PIXEL_SIZE = 3
    x_min = xmin
    y_max = ymax
    wkt_projection = proj
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1,
        gdal.GDT_Float32, )
    dataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE))  
    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.
import numpy as np
import copy

# no buffer 
#preds_test_mod = np.load('/home/cc/move/preds_test_total_PCfull.npy')
#dim = np.load('/home/cc/move/Totaldata_PCfull.npy').shape[:2]
#mask = np.load('/home/cc/move/mask_PCfull.npy')
#numr = dim[0]//224
#numc = dim[1]//224
#count = -1
#for i in range(numr):
#    for j in range(numc):
#        count += 1    
#        temp = preds_test_mod[count]
#        if j == 0:
#            rows = temp
#        else:
#            rows = np.concatenate((rows,temp),axis = 1)
#    if i == 0:
#        prediction_map = copy.copy(rows)
#    else:
#        prediction_map = np.concatenate((prediction_map,rows),axis = 0)
#prediction_map = prediction_map[:,:,0]
#prediction_map = prediction_map*mask[:prediction_map.shape[0],:prediction_map.shape[1]]

# Plot the map
#import matplotlib.pyplot as plt
#reference = np.load('/share/Stream_line_detection/data/reference.npy')
#fig= plt.figure(figsize=(16,16))
#plt.subplot(1,2,1)
#plt.title('Predicted map',fontsize = 16)
#plt.imshow(prediction_map,cmap='gray',vmin=0, vmax=255)
#plt.subplot(1,2,2)
#plt.title('Reference map',fontsize = 16)
#plt.imshow(reference,cmap='gray',vmin=0, vmax=255)
#plt.show()

# with buffer
preds_test_mod = np.load('./results/preds_test_model_augv_attention2_22082020.npy')

# #With_NAIP
# dim = np.load('./Total_data/total.npy').shape[:2]
#Without_NAIP
dim = np.load('./Total_data/total_without_NAIP.npy').shape[:2]

mask = np.load('./Total_data/mask.npy')
numr = dim[0]//(IMG_WIDTH - buf*2)
numc = dim[1]//(IMG_WIDTH - buf*2)
count = -1
for i in range(numr):
    for j in range(numc):
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
prediction_map = prediction_map*mask[:prediction_map.shape[0],:prediction_map.shape[1]]
# write out the map
proj_wkt = 'PROJCS["North_America_Albers_Equal_Area_Conic",GEOGCS["GCS_North_American_1983",DATUM["North_American_Datum_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["longitude_of_center",-96],PARAMETER["Standard_Parallel_1",20],PARAMETER["Standard_Parallel_2",60],PARAMETER["latitude_of_center",40],UNIT["Meter",1],AUTHORITY["EPSG","102008"]]'
nam = './results/organize_prediction_result_model_train_from_scratch_NAIP_100_samples_20201012-174105.tif'
array_to_raster(prediction_map,uc[0],uc[1],dim[0],dim[1],proj_wkt,nam)

