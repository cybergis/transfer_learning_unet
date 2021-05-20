# Training data augmentation
import numpy as np
import os
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
save_root="./train_test_dataset/without_NAIP/nodata_as_0/"
m = ''
# reference array
reference = np.load('./Total_data/reference_nodata_as_0.npy')
# total prediction feature maps
total = np.load('./Total_data/total_without_NAIP.npy')
# Top-left coordinates of training patches
trainlo = np.load('./train_test_dataset/without_NAIP/nodata_as_0/train_patches_top-left.npy')
# add 200 buffer pixels to each patch

print("Load data complete!")

pad = 200
trainlo[:,0] += pad
trainlo[:,1] += pad
depth = total.shape[2]
reference = np.pad(reference,(pad,pad),'symmetric')
for i in range(depth):
    temp = np.pad(total[:,:,i],(pad,pad),'symmetric')[:,:,np.newaxis]
    if i == 0:
        totaln = temp
    else:
        totaln = np.concatenate((totaln,temp),axis = 2)
def process(lpathc,lref):
    # rotate a random degree between -50 and 130
    lpathc = np.concatenate((lpathc,lref[:,:,np.newaxis]),axis = 2)
    rotate = iaa.Affine(rotate=(-50, 130))
    image1 = rotate.augment_image(lpathc)
    # rotate a random degree between 230 and 310
    rotate = iaa.Affine(rotate=(230, 310))
    image2 = rotate.augment_image(lpathc)

    #Scale a random ratio between 0.3 and 0.6
    scale = iaa.Affine(scale={"x": (0.3, 0.6), "y": (0.3, 0.6)})
    image3 = scale.augment_image(lpathc)

    #Scale a random ratio between 1.5 and 2.0
    scale = iaa.Affine(scale={"x": (1.5, 2.0), "y": (1.5, 2.0)})
    image4 = scale.augment_image(lpathc)
    
    # shear a random degree between -30 and 30
    shear = iaa.Affine(shear=(-30, 30))
    image5 = shear.augment_image(lpathc)    
    
    # flip horizontally
    flip = iaa.Fliplr(1.0)
    image6 = flip.augment_image(lpathc)  
    
    # Add Guassian noises
    #gua = iaa.AdditiveGaussianNoise(scale=(10, 20))
    #image6 = gua.augment_image(lpathc)
    #ref6 = gua.augment_image(lref)  
    oii = []
    orr = []
    for i in [image1,image2,image3,image4,image5,image6]:
        oii.append(i[pad:(pad+224),pad:(pad+224),:-1])
        orr.append(i[pad:(pad+224),pad:(pad+224),-1])
    return [oii,orr]

# Concatenate augmented training data based on different types of augmentations
pc = 0
train_data_aug = []
for i in range(len(trainlo)):
    lo = trainlo[i]
    lpatch = totaln[(lo[0]-pad):(lo[0]+224+pad),(lo[1]-pad):(lo[1]+224+pad),:]
    lref = reference[(lo[0]-pad):(lo[0]+224+pad),(lo[1]-pad):(lo[1]+224+pad)]
    if len(train_data_aug) == 0:
        train_data_aug = lpatch[pad:(-pad),pad:(-pad),:][np.newaxis,:,:,:]
        train_label_aug = lref[pad:(-pad),pad:(-pad)][np.newaxis,:,:]
    else:
        train_data_aug = np.concatenate((train_data_aug,lpatch[pad:(-pad),pad:(-pad),:][np.newaxis,:,:,:]),axis = 0)
        train_label_aug = np.concatenate((train_label_aug,lref[pad:(-pad),pad:(-pad)][np.newaxis,:,:]),axis = 0)
    [reim,rere] = process(lpatch,lref)
    for j in range(6):
        train_data_aug = np.concatenate((train_data_aug,reim[j][np.newaxis,:,:,:]),axis = 0)
        train_label_aug = np.concatenate((train_label_aug,rere[j][np.newaxis,:,:]),axis = 0)
    if i%30 == 0:
        np.save(save_root+'augmented_data/train_data_augP'+str(pc)+'.npy',train_data_aug)
        np.save(save_root+'augmented_data/train_label_augP'+str(pc)+'.npy',train_label_aug)
        train_data_aug = []
        train_label_aug = []
        pc+=1
        
# store training data after different types of augmentations
np.save(save_root+'augmented_data/train_data_augP'+str(pc)+'.npy',train_data_aug)
np.save(save_root+'augmented_data/train_label_augP'+str(pc)+'.npy',train_label_aug)

print("augmentation saved!")

# Concatenate the training data of different augmentations
for i in range(pc+1):
    temp = np.load(save_root+'augmented_data/train_data_augP'+str(i)+'.npy')
    templ = np.load(save_root+'augmented_data/train_label_augP'+str(i)+'.npy')
    if i == 0:
        fdata = temp
        fl = templ
    else:
        fdata = np.concatenate((fdata,temp),axis = 0)
        fl = np.concatenate((fl,templ),axis = 0)

# remove unnesessary intermediate files
#os.system('rm /content/drive/My Drive/USGS/Notebooks/data/train_*_augP*.npy')
# Shuffle the finalized file and save as .npy
rand = np.arange(len(fdata))
np.random.shuffle(rand)
train_data_aug = fdata[rand]
train_label_aug = fl[rand]
np.save(save_root+'augmented_data/train_data_aug'+m+'.npy',train_data_aug)
np.save(save_root+'augmented_data/train_label_aug'+m+'.npy',train_label_aug[:,:,:,np.newaxis])

print("Training data augmentation complete!")

###### visualization ########
#    import pdb
#    pdb.set_trace() 
#    plt.imshow(lref[pad:(-pad),pad:(-pad)])
#    plt.show()
#    plt.imshow(lpatch[:,:,0][pad:(-pad),pad:(-pad)])
#    plt.show()    
#    aug = ['rotate1','rotate2','scale1','scale2','shear','flip']
#    for i in range(6):
#        print(aug[i])
#        plt.imshow(rere[i])
#        plt.show()
#        plt.imshow(reim[i][:,:,0])
#        plt.show()