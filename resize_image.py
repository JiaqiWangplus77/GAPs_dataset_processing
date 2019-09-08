#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 16:14:42 2019

@author: jiaqiwang0301@win.tu-berlin.de
"""

import torch
from torchvision import transforms
import os

from PIL import Image

base_folder = os.getcwd()
image_folder = os.path.join(base_folder, 'images/images/')
proportion = 0.4
resize = (int(1080*proportion), int(1920*proportion))

file_list = [filename for filename in os.listdir(image_folder) \
                 if not filename.startswith('.')]
p = transforms.Compose([transforms.Resize(resize)])

new_image_folder = os.path.join(base_folder, 'images/images_resize/')
os.makedirs(new_image_folder, exist_ok= True)


   
for file in file_list:
    img = Image.open(os.path.join(image_folder, file))
    img2 = p(img)
    img2.save(os.path.join(new_image_folder, file)) 
    print(file, 'finished')
    




