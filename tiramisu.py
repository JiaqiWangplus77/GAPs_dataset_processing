#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:16:18 2019

@author: jiaqiwang0301@win.tu-berlin.de
convert the image to segmentation type.
merge the results (transparent label on the images)
"""


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

deal_with_all_files = 0
# if true, deal with all files of chosen type
# if false, deal with single file
current_type_choice = 2 # 0:train,1:test,2 valid
image_num = 0  # raise error when out of range
bbox_padding = 0  
# the original bbox is 64*64, make it larger to merge better
bbox_merge_padding = 0
# make the merged bbox larger for further merging 
radius_minus= 6

# base folder is GAPs folder
base_folder = os.getcwd()
# define the folder to extracct images and labels
image_folder = os.path.join(base_folder, 'GAPS_v1/images/') 
label_folder = os.path.join(base_folder, 'GAPS_v1//labels/') 

# create new folders to save the label files
image_folder_name = os.path.join('images/tiramisu/tiramisu_resize7_2_annotation_result_merged/')
os.makedirs(image_folder_name,exist_ok= True)

tiramisu_folder_name = os.path.join('images/tiramisu/tiramisu_resize7_2_annotation/')

# 0:intact_road, 1:applied_patch, 2:pothole, 3:inlaid_patch, 4:open_joint, 5:crack
data_type = [0,1,2,3,4,5]

# input part finished

num_typ = [1418, 500, 50]    # [1418, 500, 50] 

    
if deal_with_all_files:
    image_num = 0
    plot_num = range(num_typ[current_type_choice])
else:
    plot_num = [image_num]
    
# the number of photos for each type
if image_num-1 > num_typ[current_type_choice]:
    raise Exception('out of range')
    

image_typ = ['train', 'test', 'valid']
typ_name_tiramisu = ['train', 'test', 'val']
current_image_type = image_typ[current_type_choice]

annot_folder_name = os.path.join(base_folder,
                                 tiramisu_folder_name,
                                 typ_name_tiramisu[current_type_choice]+'annot/')

final_image_folder = os.path.join(base_folder,
                                  tiramisu_folder_name,
                                  typ_name_tiramisu[current_type_choice]+'/')
os.makedirs(annot_folder_name,exist_ok= True)
os.makedirs(final_image_folder,exist_ok= True)

# define the name format(pure name or name with folder) for photo and txt file
imag_name = current_image_type + '_{:04d}.jpg'
imag_name2 = current_image_type + '_{:04d}.png'
label_name = current_image_type + '_{:04d}.txt'
image_template = image_folder + imag_name
label_template = label_folder + label_name

radius = 32 - radius_minus

def new_dataset_num(data_type):
    return np.arange(len(data_type))

new_dataset_num = new_dataset_num(data_type)

intact_road = (0,0,0)  # black
applied_patch = (0,255,0)  # green
pothole = (100,0,255)  # blue
inlaid_patch = (255,255,255)   # white
open_joint = (204,0,255) #purple
crack = (255,0,0) # red
colors = [intact_road,applied_patch, pothole, inlaid_patch, open_joint, crack]


for image_num in plot_num:
    image_filename = image_template.format(image_num)
    label_filename = label_template.format(image_num)
    
    image = np.array(Image.open(image_filename))
    image = np.stack((image,)*3, axis=-1)
    
    #image = cv2.imread(image_filename)
    imag_width, imag_height = image.shape[1], image.shape[0]
    #image2 = image.copy()
    empty_image1 = np.ones_like(image) * 255
    empty_image2 = np.ones_like(image) * 255

    if not deal_with_all_files:
        plt.figure()
        plt.imshow(image)   
        plt.title('original image: '+ imag_name.format(image_num))
    
    if not os.path.isfile(label_filename):  
        pass
        #cv2.imwrite(annot_folder_name+imag_name2.format(image_num),np.zeros([imag_width, imag_height]))        

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
#            # compared with rectangel bounding box
#            coordinate = np.array([imag_width * obj[index,1]-32-bbox_padding, 
#                                   imag_height * obj[index,2]-32-bbox_padding,
#                                   imag_width * obj[index,1]+32+bbox_padding, 
#                                   imag_height * obj[index,2]+32+bbox_padding]).astype(int).T
#           
#            for i in range(len(coordinate)):
#                bbox=[(coordinate[i,0],coordinate[i,1]),(coordinate[i,2],coordinate[i,3])]
#                cv2.rectangle(image, pt1=bbox[0], pt2=bbox[1],color=colors[item],thickness=-1) 
             
            # compared with circle bounding box
            coordinate = np.array([imag_width * obj[index,1], 
                                   imag_height * obj[index,2]
                                   ]).astype(int).T
            # draw bbox, fill the bounding box with color
            data_num = int(new_dataset_num[item])
            labels = np.ones([imag_height,imag_width])
            for i in range(len(coordinate)):
                bbox=[(coordinate[i,0],coordinate[i,1])]
                cv2.circle(labels, center=bbox[0], radius=radius, color=(data_num),thickness=-1) 
                cv2.circle(empty_image2, center=bbox[0], radius=radius, color=colors[item],thickness=-1)  
#            if not deal_with_all_files:
#                plt.figure()
#                plt.imshow(image) 
#                plt.title("after circle")                             
            

            
        #image_combine = np.concatenate((image_original,image,image2), axis=1)    
        w_img = 0.8
        w_label = 0.2
            
        image_merged = cv2.addWeighted(image,w_img,empty_image2,w_label,0)
        if not deal_with_all_files:
            plt.figure()
            plt.imshow(image_merged)
            plt.title('circle bounding box')  
            
            
#            plt.figure()
#            plt.imshow(image2)  
#            plt.title('original rectangle bounding box')
#            plt.figure()
#            plt.imshow(image_combine)  
#            plt.title('image combination')            
            
            
#        cv2.imwrite(image_folder_name+imag_name2.format(image_num),image_merged)
#        cv2.imwrite(annot_folder_name+imag_name2.format(image_num),labels)
        
        print(imag_name2.format(image_num), ' finished')
#        

