#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:18:43 2019

@author: jiaqiwang0301@win.tu-berlin.de

count the number for each dataset for training ,test and valid
and count the total number for the whole dataset
"""

from gaps_dataset import gaps
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# choose image type
current_type_choice = 2 # 0,1,2


label_folder = '/images/bbox_merge/labels_with_crack_only/'
 

image_typ = ['train', 'test', 'valid']

num_typ = [1418, 500, 50] 
destination_directory = os.getcwd()

np.set_printoptions(suppress=True)


num_each_dataset = np.zeros([3,5])

for current_type_choice in range(3):
    current_image_type = image_typ[current_type_choice]
    label_template = destination_directory + label_folder + current_image_type + '_{:04d}.txt'

    for image_num in range(num_typ[current_type_choice]):
        label_filename = label_template.format(image_num)

        if not os.path.isfile(label_filename):    
            pass
        else:
            obj = np.loadtxt(label_filename).reshape(-1,5)
            num = np.array(obj[:,0],dtype=np.int8)
            for i in num:
                num_each_dataset[current_type_choice,i] += 1 
    print(current_image_type,num_each_dataset[current_type_choice,:])
                
print('total number of each class \n', num_each_dataset.sum(axis=0))
        

    
    




# color for each class
#intact_road = (0,0,0)
#applied_patch = (0,255,0)
#pothole = (0,0,255)
#inlaid_patch = (255,255,255)
#open_joint = (255,0,0)
#crack = (128,128,0)
#colors = [intact_road, applied_patch, pothole, inlaid_patch, open_joint, crack]


#yolo_width = round(64 / imag_width, 6)
#yolo_height = round(64 / imag_height, 6)


#
#
##%%    
#patch_ref_distress = patch_ref[np.argwhere(patch_ref[:,4]==1)[:,0],:]
#index = np.argsort(patch_ref_distress[:,0])
#patch_ref_distress = patch_ref_distress[index,:]
#yolo_table_whole = np.array([patch_ref_distress[:,0],
#                             patch_ref_distress[:,-1],
#                             np.round((patch_ref_distress[:,2]+32)/imag_width, 5),
#                             np.round((patch_ref_distress[:,1]+32)/imag_height, 5),
#                             yolo_width*np.ones_like(patch_ref_distress[:,0]),
#                             yolo_height*np.ones_like(patch_ref_distress[:,0])
#                             ]).T
#
#
#
##%%
## sort the information with the index of image
#current_index = int(yolo_table_whole[0][0])
#yolo_table_whole = yolo_table_whole.tolist()
#
#num_yolo_info = len(yolo_table_whole)
#i = 0
#image_infos = []
#single_object_info = []
#
#
#
#while i < num_yolo_info:    
#    current_item = yolo_table_whole.pop(0)
#    if int(current_item[0]) == current_index:
#        single_object_info.append(current_item)
#    else:
#        current_index = current_item[0]
#        image_infos.append(single_object_info)
#        single_object_info = []
#    i += 1
#
##%%
## create folder label and generate txt file to store the object information
## the format is the same with yolo v3 label.the name is the same with image
#os.makedirs('images/labels',exist_ok= True) 
#base_path = os.getcwd() + '/images/labels/'
#
#image_name = current_image_type + '_{:04d}.jpg'
#
#def string_generate(list1):
#    str1=""
#    for item in list1:
#        str1 += str(item) + " "
#    return str1
#
#for i in range(len(image_infos)):
#    imag = image_infos[i]
#    if len(imag) == 0:
#        continue
#    image_filename = image_name.format(int(imag[0][0])).split('.')[0] + '.txt'
#    f = open(base_path + image_filename,'w')    
#    for j in range(len(imag)):        
#        object_info = string_generate(imag[j][1:])          
#        f.write(object_info +'\n')
#    f.close()
    
#imag_index = np.argwhere(patch_ref_distress[:,0]==3)[:,0]
## since already sorted? still necessary to find the index?
#imag_3 = patch_ref_distress[imag_index,:]




## plot with bounding box
#image_filename = image_template.format(3)
#image = cv2.imread(image_filename)
#
#for i in range(imag_3.shape[0]):
#    col, row =imag_3[i,2], imag_3[i,1]
#    bbox=[(col, row ), (col + 64, row + 64)] 
#    cv2.rectangle(image, pt1=bbox[0], pt2=bbox[1],color=(255,0,0),thickness=5)
#    
#plt.imshow(image)   
#cv2.imwrite('test.jpg',image)


#%%
# test for plot
#patch_ref_sort = patch_ref[np.argsort(patch_ref[:,0]),:]
#imag_index = np.argwhere(patch_ref_sort[:,0]==0)[:,0]
#imag_0 = patch_ref_sort[imag_index,:]
#
#
## plot with bounding box
#image_filename = image_template.format(0)
#image = cv2.imread(image_filename)
#
#imag = imag_0
#for i in range(imag.shape[0]):
#    col, row, cls_label = imag[i,2], imag[i,1], imag[i,5]
#    bbox=[(col, row ), (col + 64, row + 64)] 
#    cv2.rectangle(image, pt1=bbox[0], pt2=bbox[1],color=colors[cls_label],thickness=5)
#    
#plt.imshow(image)   
#cv2.imwrite('test0.jpg',image)
    
