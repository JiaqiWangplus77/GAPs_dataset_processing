#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:12:33 2019

@author: jiaqiwang0301@win.tu-berlin.de
plot from yolo files
because sometimes there is something wrong with the data
so adjust the data and check the result
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

deal_with_all_files = 0
# if true, deal with all files of chosen type
# if false, deal with single file

image_num = 0# raise error when out of range


# define the folder to extracct images and labels
#temp = 'modify_coord'
#class_info = os.path.join('YOLO','image_classification',temp)

#image_folder = os.path.join('tiramisu_dataset','images')
image_folder = os.path.join('YOLO',
                            'filtered_dataset_multi_classes',
                            'augmentation/class3/images')
yolo_label_folder = os.path.join('YOLO','filtered_dataset_multi_classes',
                                 'augmentation/class3/labels')

result_folder = os.path.join('YOLO','filtered_dataset_multi_classes',
                             'augmentation/class3/annotation_result') 
os.makedirs(result_folder, exist_ok= True)

#label_modified_folder = os.path.join('YOLO',temp+'_modified') 
#os.makedirs(label_modified_folder, exist_ok= True)

file_name = sorted([file for file in os.listdir(yolo_label_folder)
                  if file.endswith('txt')])

#file_name = ['train_0541_1_1.jpg']


if deal_with_all_files:
    image_num = 0
    plot_num = range(len(file_name))
else:
    plot_num = [image_num]

def coordinate_transform(yolo, imag_width, imag_height):
    """
    transfrom from yolo format [type, x_center,y_center,width,height] (scaled to 1)
    to the coordinate of the two corners, which makes it easier for bounding box plotting"""
     
    coord = np.zeros_like(yolo)
    coord[:,0] = yolo[:,0] 
    coord[:,1] = (yolo[:,1] - yolo[:,3] / 2) * imag_width
    coord[:,2] = (yolo[:,2] - yolo[:,4] / 2) * imag_height
    coord[:,3] = (yolo[:,1] + yolo[:,3] / 2) * imag_width
    coord[:,4] = (yolo[:,2] + yolo[:,4] / 2) * imag_height
    coord = np.int64(coord)    
    return coord

color = [(255,0,0),(0,255,0),(0,0,255)]

for image_num in plot_num:

    name = file_name[image_num].split('.')[0]
    
    
    #name = 'train_0917_1_641'

    image_name = name + '.jpg'
    label_name = name + '.txt'
    image_filename = os.path.join(image_folder, image_name)
    image = np.array(Image.open(image_filename))
    image2 = np.array(Image.open(image_filename))
    label_filename = os.path.join(yolo_label_folder,
                                  label_name)   
    if not os.path.isfile(image_filename):
        print('passed')
    else:
        yolo = np.array(np.loadtxt(label_filename)).reshape(-1,5) 
       
        imag_height, imag_width = image.shape[0],image.shape[1]  
        coord = coordinate_transform(yolo,imag_width,imag_height)       
        for i in range(coord.shape[0]):
            bbox = [(coord[i,1],coord[i,2]),(coord[i,3],coord[i,4])]            
            cv2.rectangle(image, pt1=bbox[0], pt2=bbox[1],color=color[coord[i,0]],thickness=2)
        if not deal_with_all_files:
            plt.figure()   
            plt.imshow(image)
       
        cv2.imwrite(os.path.join(result_folder,image_name),image)
        
        

      
#        y = yolo          
# 
    #       modifity the data and show the result       
#        y = np.array([[2.    , 0.5 , 0.4157, 1, 0.0907]])
#        coord = coordinate_transform(y,imag_width,imag_height)
#        for i in range(coord.shape[0]):
#            bbox = [(coord[i,1],coord[i,2]),(coord[i,3],coord[i,4])]            
#            cv2.rectangle(image2, pt1=bbox[0], pt2=bbox[1],color=color[coord[i,0]],thickness=2)       
#        #cv2.imwrite(os.path.join(result_folder,name),image2)
#        plt.figure(figsize=[8,8])  
#        plt.imshow(image2)
#        plt.title(name+' after modification')

        '''
        np.savetxt(os.path.join('YOLO/filtered_dataset_multi_classes/',label_name),
                   y,fmt='%s',newline='\n',delimiter=' ') 
        '''
