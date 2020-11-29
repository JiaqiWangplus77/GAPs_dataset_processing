#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jiaqiwang0301@win.tu-berlin.de

generate YOLO files from tiramisu label 
with crop_clear_crack dataset

"""


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


deal_with_all_files = 0
# if true, deal with all files of chosen type
# if false, deal with single file
#current_type_choice = 2 # 0:train,1:test,2 valid
image_num = 10  # raise error when out of range
bbox_padding = 0 
# the original bbox is 64*64, make it larger to merge better
bbox_merge_padding = 3
# make the merged bbox larger for further merging 
image_area_limit = 4500

#image_typ = ['train', 'test', 'val']
#current_image_type = image_typ[current_type_choice]
bbox_folder = os.path.join('YOLO', 'image_with_bbox')
os.makedirs(bbox_folder,exist_ok= True)

# define the folder to extracct images and labels
image_folder = os.path.join('tiramisu_dataset','images')
label_folder = os.path.join('tiramisu_dataset','labels2')

# create new folders to save the label files
result_folder_name = os.path.join('YOLO','merge_process')
os.makedirs(result_folder_name,exist_ok= True)

## 0:intact_road, 1:applied_patch, 2:pothole, 3:inlaid_patch, 4:open_joint, 5:crack
#data_type = [1,2,3,4,5]

#num_typ = [1418, 500, 50]    # [1418, 500, 50] 
imag_name = sorted([file for file in os.listdir(image_folder)])

label_name = [file for file in os.listdir(label_folder)]
    
# the number of photos for each type
#if image_num-1 > num_typ[current_type_choice]:
#    raise Exception('out of range')
    
if deal_with_all_files:
    image_num = 0
    plot_num = range(len(imag_name))

else:
    plot_num = [image_num]


def new_dataset_num(data_type):
    return np.arange(len(data_type))

#new_dataset_num = new_dataset_num(data_type)


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
        cv2.rectangle(img, pt1=bbox[0], pt2=bbox[1],color=[255,255,255],thickness=-11)
        #cv2.rectangle(img, pt1=bbox[0], pt2=bbox[1],color=[255,0,0],thickness=5) 
        
#    plt.figure()
#    plt.imshow(img) 
#    cv2.imwrite(os.path.join(bbox_folder,name),img)
        #plt.title('draw contours'+str(rect)+str(i))  
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
times =[]
for image_num in plot_num:
    if image_num % 100 == 0:
        print(image_num, ' files finished')
    name = imag_name[image_num]
    #name = 'train_0420_1_641.jpg'
    image_filename = os.path.join(image_folder,
                                  name)
    label_filename = os.path.join(label_folder,
                                  name.replace('jpg','png'))
    image = np.array(Image.open(image_filename))
    
    if not os.path.isfile(label_filename):    
        pass
    else:
        label = np.array(Image.open(label_filename)) * 255
        label = np.stack((label,)*3, axis=-1) 
        empty_img = np.zeros([label.shape[0],label.shape[1],3],dtype=np.uint8)
        imag_width = image.shape[1]
        imag_height = image.shape[0]        
                          
        i = 0
        
        contours,hierarchy = find_contours(label)                
        empty_img = draw_contours(empty_img,contours,-1) 
        num_contours_new = len(contours)
#        plt.figure()
#        plt.imshow(empty_img)
        
        while True:
        # find the rectangle to cover the contours
            i += 1
            num_contours_old = num_contours_new            
            contours,hierarchy = find_contours(empty_img)                
            empty_img = draw_contours(empty_img,contours,-1)           
            #contours = delete_child_contours(contours,hierarchy)
            
            num_contours_new = len(contours)
            
            if num_contours_new == num_contours_old:              
                break
           
        #img = np.zeros([imag_height,imag_width,3],dtype=np.uint8)
        i = i + 1

        rectangle=[]
        yolo_table_whole = []
        contours = contours[::-1] #!!!
        for num in range(len(contours)):                
            cnt = contours[num]
            rect = cv2.boundingRect(cnt)
                       
#            if rect[2] * rect[3] < image_area_limit:
#                print(name)
#                
#                continue
  
            #print(rect)
            rectangle.append(rect)
            bbox=[(rect[0],
                   rect[1]),
                   (rect[0]+rect[2]-i*bbox_merge_padding,
                    rect[1]+rect[3]-i*bbox_merge_padding)]                
            cv2.rectangle(image, pt1=bbox[0], pt2=bbox[1],color=[255,0,0],thickness=5)            
            rect = list(rect)
            #data_num = new_dataset_num[item]
            rect.insert(0,int(0))
            yolo_table_whole.append(rect)
                
#                plt.figure()
#                plt.imshow(img)  rectangle[:,2]=rectangle[:,2]+rectangle[:,0]

                       
        if not deal_with_all_files:
            plt.figure()
            plt.imshow(image)
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
#        np.savetxt(os.path.join(result_folder_name,name.replace('jpg','txt')),
#                   yolo_table_whole,fmt='%s',newline='\n',delimiter=' ')
#        cv2.imwrite(os.path.join(bbox_folder,name),image)
        #print(label_name.format(image_num),' finished')
