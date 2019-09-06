#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:16:18 2019

@author: jiaqiwang0301@win.tu-berlin.de
convert the image to segmentation type
"""


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


deal_with_all_files = 0
# if true, deal with all files of chosen type
# if false, deal with single file
current_type_choice = 1 # 0:train,1:test,2 valid
image_num = 5  # raise error when out of range
bbox_padding = 0  
# the original bbox is 64*64, make it larger to merge better
bbox_merge_padding = 0
# make the merged bbox larger for further merging 
radius_minus= 20

# base folder is GAPs folder
base_folder = os.getcwd()
# define the folder to extracct images and labels
image_folder = base_folder + '/images/images/'
label_folder = base_folder + '/images/labels/'
# create new folders to save the label files
label_folder_name = 'images/tiramisu_annotation/'
label_folder_name2 = 'images/tiramisu_annotation_result/'

# 0:intact_road, 1:applied_patch, 2:pothole, 3:inlaid_patch, 4:open_joint, 5:crack
data_type = [1,2,3,4,5]

label_folder_merge = base_folder + '/' + label_folder_name
#os.makedirs('images/bbox_merge',exist_ok= True)
os.makedirs(label_folder_name, exist_ok= True)
os.makedirs(label_folder_name2, exist_ok= True)

num_typ = [1418, 500, 50]    # [1418, 500, 50] 
# the number of photos for each type
if image_num-1 > num_typ[current_type_choice]:
    raise Exception('out of range')
    
if deal_with_all_files:
    image_num = 0
    plot_num = range(num_typ[current_type_choice])
else:
    plot_num = [image_num]
    

image_typ = ['train', 'test', 'valid']
current_image_type = image_typ[current_type_choice]


# define the name format(pure name or name with folder) for photo and txt file

imag_name = current_image_type + '_{:04d}.jpg'
imag_name2 = current_image_type + '_{:04d}.png'
label_name = current_image_type + '_{:04d}.txt'
image_template = image_folder + imag_name
label_template = label_folder + label_name


radius = 64 - radius_minus

def new_dataset_num(data_type):
    return np.arange(len(data_type))

new_dataset_num = new_dataset_num(data_type)

intact_road = (0,0,0)  # black
applied_patch = (0,255,0)  # green
pothole = (0,0,255)  # blue
inlaid_patch = (255,255,255)   # white
open_joint = (204,0,255) #purple
crack = (255,0,0) # red
colors = [intact_road,applied_patch, pothole, inlaid_patch, open_joint, crack]

def find_contours(img):

    # convert the image to single channel and then convert to binary image 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    ret,thresh = cv2.threshold(img,127,255,0)
    # find the counters
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours,hierarchy


for image_num in plot_num:
    image_filename = image_template.format(image_num)
    label_filename = label_template.format(image_num)
    
    image = cv2.imread(image_filename)
    imag_width = image.shape[1]
    imag_height = image.shape[0]

    img = np.zeros([imag_height,imag_width,3],dtype=np.uint8)
    img2 = np.zeros([imag_height,imag_width,3],dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if not os.path.isfile(label_filename):    
        pass
    else:
        yolo_table_whole = []
        obj = np.loadtxt(label_filename).reshape(-1,5) # when there is only 1 object
    #        obj_class = np.unique(obj[:,0])
        obj_class = data_type
    
        for item in range(len(obj_class)):
            #plt.figure()
            #img = np.zeros([imag_height,imag_width,3],dtype=np.uint8)
            index = np.array(np.where(obj[:,0] == data_type[item]))[0,:]   
            if index.shape[0] == 0:
                continue

            coordinate = np.array([imag_width * obj[index,1], 
                                   imag_height * obj[index,2]
                                   ]).astype(int).T
            # draw bbox, fill the bounding box with color
            data_num = int(new_dataset_num[item])
            for i in range(len(coordinate)):
                bbox=[(coordinate[i,0],coordinate[i,1])]
                cv2.circle(img, center=bbox[0], radius=radius, color=(data_num),thickness=-1) 
                cv2.circle(img2, center=bbox[0], radius=radius, color=colors[item],thickness=-1)   
        
        if not deal_with_all_files:
            plt.figure()
            plt.imshow(image)  
            
        cv2.imwrite(label_folder_name+imag_name2.format(image_num),img)
        cv2.imwrite(label_folder_name2+imag_name2.format(image_num),image)
        print(imag_name2.format(image_num), ' finished')
        

