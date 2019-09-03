#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:18:43 2019

@author: jiaqiwang0301@win.tu-berlin.de

choose the image and plot the image with bounding box
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# choose image type
deal_with_all_files = 1
# if true, deal with all files of chosen type
# if false, deal with single file
current_type_choice = 2 # %N 0:train,1:test,2 valid
image_num = 45

base_folder = os.getcwd()
# define the folder to extracct images and labels
image_folder = base_folder + '/images/images/'
label_folder = base_folder + '/images/labels/'

# define the folder to save images
img_folder = 'images/'


os.makedirs(img_folder,exist_ok=True)
image_folder_merge = base_folder + '/' + img_folder
    
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



imag_name=current_image_type + '_{:04d}.jpg'
label_name = current_image_type + '_{:04d}.txt'
image_template = image_folder + imag_name
label_template = label_folder + label_name



# color for each class
intact_road = (0,0,0)  # black
applied_patch = (0,255,0)  # green
pothole = (0,0,255)  # blue
inlaid_patch = (255,255,255)   # white
open_joint = (204,0,255) #purple
crack = (255,0,0) # red
colors = [intact_road,applied_patch, pothole, inlaid_patch, open_joint, crack]

for image_num in plot_num:
    image_filename = image_template.format(image_num)
    label_filename = label_template.format(image_num)
    
    image = cv2.imread(image_filename)
    imag_width = image.shape[1]
    imag_height = image.shape[0]
    
    
    # draw the image with bbox, if there is no problem with the surface, which means no txt file,
    # only plot the image
    
    if not os.path.isfile(label_filename):
        plt.figure()
        plt.imshow(image)
        plt.close()
    else:
        obj = np.loadtxt(label_filename).reshape(-1,5) # when there is only 1 object
        coordinate = np.array([imag_width * (obj[:,1] - obj[:,3]/2), 
                               imag_height * (obj[:,2] - obj[:,4]/2),
                               imag_width * (obj[:,1] + obj[:,3]/2), 
                               imag_height * (obj[:,2] + obj[:,4]/2)]).astype(int).T
        for i in range(len(coordinate)):
            bbox=[(coordinate[i,0],coordinate[i,1]),(coordinate[i,2],coordinate[i,3])]
            cls_label=int(obj[i][0])
            cv2.rectangle(image, pt1=bbox[0], pt2=bbox[1],color=colors[cls_label],thickness=2)              
        #plt.imshow(image)  
       
        plt.imsave(image_folder_merge+imag_name.format(image_num),image) 
        
        if not deal_with_all_files:
            plt.figure()
            plt.imshow(image)        
        print(image_num)     
    

