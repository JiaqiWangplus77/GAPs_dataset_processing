"""
Created on Fri Aug  9 19:18:43 2019

@author: jiaqiwang0301@win.tu-berlin.de

process npy file. 
generate .txt file for yolo v3
"""

from gaps_dataset import gaps
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


image_typ = ['train', 'test', 'valid']
current_type_choice = 2 # 0,1,2
current_image_type = image_typ[current_type_choice]
# num_typ = np.array([1418, 500, 51])
# patch_info= np.array([4897699,810704, 201483])

# get the patch information from .npy file
destination_directory = os.getcwd()
ref_filename = destination_directory + '/images/patch_references_{}.npy'.format(current_image_type)
patch_ref = np.load(ref_filename).astype(int)

# get the name list for train, test or valid under images/images folder
image_template = destination_directory + '/images/images/' + current_image_type + '_{:04d}.jpg'
image_dir = destination_directory + '/images/images/'
file_list = [image_dir + filename for filename in os.listdir(image_dir) \
             if not filename.startswith('.')]

image_filename = image_template.format(0)
image = cv2.imread(image_filename)
imag_width = image.shape[1]
imag_height = image.shape[0]
yolo_width = round(64 / imag_width, 6)
yolo_height = round(64 / imag_height, 6)
#%% 
# filter intact_road 
if current_image_type == 0:
    print('begin to filter the information,for train set please have some patience')  
patch_ref_distress = patch_ref[np.argwhere(patch_ref[:,4]==1)[:,0],:]
index = np.argsort(patch_ref_distress[:,0])
patch_ref_distress = patch_ref_distress[index,:]

## to generate all the labels
#index = np.argsort(patch_ref[:,0])
#patch_ref_distress = patch_ref[index,:]

yolo_table_whole = np.array([patch_ref_distress[:,0],
                             patch_ref_distress[:,-1],
                             np.round((patch_ref_distress[:,2]+32)/imag_width, 5),
                             np.round((patch_ref_distress[:,1]+32)/imag_height, 5),
                             yolo_width*np.ones_like(patch_ref_distress[:,0]),
                             yolo_height*np.ones_like(patch_ref_distress[:,0])
                             ]).T



#%%
# sort the information with the index of image
current_index = int(yolo_table_whole[0][0])
yolo_table_whole = yolo_table_whole.tolist()
num_yolo_info = len(yolo_table_whole)
i = 0
image_infos = []
single_object_info = []

while i < num_yolo_info:    
    current_item = yolo_table_whole.pop(0)
    if int(current_item[0]) == current_index:
        single_object_info.append(current_item)
    else:
        current_index = current_item[0]
        image_infos.append(single_object_info)
        single_object_info = []
    i += 1

    

#%%
# create folder label and generate txt file to store the object information
# the format is the same with yolo v3 label.the name is the same with image
label_folder = 'images/labels/'
os.makedirs(label_folder,exist_ok= True) 
base_path = os.getcwd() + '/' + label_folder
#os.makedirs('images/labels',exist_ok= True) 
#base_path = os.getcwd() + '/images/labels/'

image_name = current_image_type + '_{:04d}.jpg'

def string_generate(list1):
    str1=""
    for item in list1:
        str1 += str(item) + " "
    return str1

print('begin to generate yolo file')
for i in range(len(image_infos)):
    imag = image_infos[i]
    if len(imag) == 0:
        continue
    image_filename = image_name.format(int(imag[0][0])).split('.')[0] + '.txt'
    f = open(base_path + image_filename,'w')    
    for j in range(len(imag)):        
        object_info = string_generate(imag[j][1:])          
        f.write(object_info +'\n')
    f.close()
    print(image_filename, ' done')

print('finished')
    