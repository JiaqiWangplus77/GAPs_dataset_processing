#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:18:43 2019

@author: jiaqiwang0301@win.tu-berlin.de

merge the small fixed bbox into a bigger one 
and generate txt file with yolo format
"""


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


deal_with_all_files = 0
# if true, deal with all files of chosen type
# if false, deal with single file
current_type_choice = 2 # 0:train,1:test,2 valid
image_num = 4  # raise error when out of range
bbox_padding = 0  
# the original bbox is 64*64, make it larger to merge better
bbox_merge_padding = 1
# make the merged bbox larger for further merging 

# base folder is GAPs folder
base_folder = os.getcwd()
# define the folder to extracct images and labels
image_folder = base_folder + '/images/images/'
label_folder = base_folder + '/images/labels/'
# create new folders to save the label files
label_folder_name = 'images/bbox_merge/labels/'

# 0:intact_road, 1:applied_patch, 2:pothole, 3:inlaid_patch, 4:open_joint, 5:crack
data_type = [5]

label_folder_merge = base_folder + '/' + label_folder_name
#os.makedirs('images/bbox_merge',exist_ok= True)
os.makedirs(label_folder_name, exist_ok= True)


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
label_name = current_image_type + '_{:04d}.txt'
image_template = image_folder + imag_name
label_template = label_folder + label_name


image_filename = image_template.format(image_num)
label_filename = label_template.format(image_num)

image = cv2.imread(image_filename)
imag_width = image.shape[1]
imag_height = image.shape[0]

img = np.zeros([imag_height,imag_width,3],dtype=np.uint8)

def new_dataset_num(data_type):
    return np.arange(len(data_type))

new_dataset_num = new_dataset_num(data_type)


def find_contours(img):

    # convert the image to single channel and then convert to binary image 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    ret,thresh = cv2.threshold(img,127,255,0)
    # find the counters
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours,hierarchy

def draw_contours(img,contours,thickness):
    
    for i in range(len(contours)):
        cnt = contours[i]
        #rect = cv2.minAreaRect(cnt) #!! with rotation
        #box = np.int0(cv2.boxPoints(rect))
        #cv2.drawContours(img, [box], 0, (255, 0, 0), 5)
       
        rect = cv2.boundingRect(cnt)
        bbox=[(rect[0] - bbox_merge_padding,
               rect[1] - bbox_merge_padding),
              (rect[0] + rect[2] + bbox_merge_padding,
               rect[1]+rect[3] + bbox_merge_padding)]                
        cv2.rectangle(img, pt1=bbox[0], pt2=bbox[1],color=[255,255,255],thickness=-1)
        #cv2.rectangle(img, pt1=bbox[0], pt2=bbox[1],color=[255,0,0],thickness=5) 
#        plt.figure()
#        plt.imshow(img) 
#        plt.title('draw contours'+str(rect)+str(i))  
        #print('draw contours'+str(rect))
    return img

def delete_child_contours(contours,hierarchy):
    # check if there is contours found
    if hierarchy is None:
        return contours
    
    index = np.where(hierarchy[:,:,3]!=-1)[1]
    for ind in index[::-1]:
        del contours[ind]
        
    return contours

for image_num in plot_num:
    if image_num % 100 == 0:
        print(image_num)
    image_filename = image_template.format(image_num)
    label_filename = label_template.format(image_num)
    if not os.path.isfile(label_filename):    
        pass
    else:
        yolo_table_whole = []
        obj = np.loadtxt(label_filename).reshape(-1,5) # when there is only 1 object
#        obj_class = np.unique(obj[:,0])
        obj_class = data_type
        
        for item in range(len(obj_class)):
            #plt.figure()
            img = np.zeros([imag_height,imag_width,3],dtype=np.uint8)
            index = np.array(np.where(obj[:,0] == data_type[item]))[0,:]   
            if index.shape[0] == 0:
                continue
            coordinate = np.array([imag_width * obj[index,1]-32-bbox_padding, 
                                   imag_height * obj[index,2]-32-bbox_padding,
                                   imag_width * obj[index,1]+32+bbox_padding, 
                                   imag_height * obj[index,2]+32+bbox_padding]).astype(int).T
            # draw bbox, fill the bounding box with color
            for i in range(len(coordinate)):
                bbox=[(coordinate[i,0],coordinate[i,1]),(coordinate[i,2],coordinate[i,3])]
                cv2.rectangle(img, pt1=bbox[0], pt2=bbox[1],color=[255,255,255],thickness=-1)              
                        
            i = 0
            num_contours_new = 1

            while True:
            # find the rectangle to cover the contours
                i += 1
                num_contours_old = num_contours_new
                
                contours,hierarchy = find_contours(img)                
                img = draw_contours(img,contours,-1)           
                contours = delete_child_contours(contours,hierarchy)
                
                num_contours_new = len(contours)
                
                if num_contours_new == num_contours_old:              
                    break
           
            #img = np.zeros([imag_height,imag_width,3],dtype=np.uint8)
            i = i + 1
            rectangle=[]
            for num in range(len(contours)):                
                cnt = contours[num]
                rect = cv2.boundingRect(cnt)
                rectangle.append(rect)
                #print(rect)
                
                bbox=[(rect[0],
                       rect[1]),
                       (rect[0]+rect[2]-i*bbox_merge_padding,
                        rect[1]+rect[3]-i*bbox_merge_padding)]                
                cv2.rectangle(img, pt1=bbox[0], pt2=bbox[1],color=[255,0,0],thickness=5)            
                rect = list(rect)
                data_num = new_dataset_num[item]
                rect.insert(0,int(data_num))
                yolo_table_whole.append(rect)
                
#                plt.figure()
#                plt.imshow(img)  rectangle[:,2]=rectangle[:,2]+rectangle[:,0]

            
            
            if not deal_with_all_files:
                plt.figure()
                plt.imshow(img)
            """
            convert the x,y,w,h to x_center,y_center,width,height. 
            and scale to [0,1]
            """        
        yolo_table_whole = np.array(yolo_table_whole, dtype=np.float64)
        if yolo_table_whole.size == 0:
           continue 
            
        yolo_table_whole[:,1] = np.round((yolo_table_whole[:,1] + yolo_table_whole[:,3]/2)/imag_width, 4)
        yolo_table_whole[:,2] = np.round((yolo_table_whole[:,2] + yolo_table_whole[:,4]/2)/imag_height, 4)
        yolo_table_whole[:,3] = np.round(yolo_table_whole[:,3]/imag_width, 4)      
        yolo_table_whole[:,4] = np.round(yolo_table_whole[:,4]/imag_height, 4)
        
        
        #filename_label = image_filename.split('/')[-1].split('.')[0] + '.txt'
        np.savetxt(label_folder_name+label_name.format(image_num),yolo_table_whole,fmt='%s',newline='\n',delimiter=' ')
