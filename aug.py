import cv2
import os
import glob
import pathlib
import random
import numpy as np
import albumentations as A
import torchvision.transforms.functional as TF

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

#Create Augmentations, with p = probability of being applied
transform = A.Compose([
	A.RandomCrop(width = 150, height = 150, p = 0.5),
	A.HorizontalFlip(p = 0.5),
	A.VerticalFlip(p = 0.5),
], p = 1.0)

#change to whatever directory -> DIRECTORIES
#Make sure augmented_images and augmented_masks folders exist in data, and data folder is in same dir as aug.py
script_dir = 'C:/Users/jonat/Desktop/Assignment Garbage/COMPSCI760/Fire_Mapping_Project/Fire_Mapping_Project-main'
image_dir = script_dir + "/data/train_images" 
mask_dir = script_dir + "/data/train_masks"
target_dir = script_dir + "/data/augmented_images"
target_masks = script_dir + "/data/augmented_masks"

#Initialize and print variables
image_list = []
mask_list = []
print(script_dir)
print(image_dir)
print(mask_dir)

#Read Images
for i in glob.iglob(image_dir + '/*'):
	image_list.append(np.asarray(Image.open(i)))
for j in glob.iglob(mask_dir + '/*'):
	mask_list.append(np.asarray(Image.open(j)))

image_list = np.array(image_list, dtype = object)
mask_list = np.array(mask_list, dtype = object)

#print(len(image_list))

#iterate through entire image list to transform both image and mask, before saving to augmented images/masks folder
for i in range(len(image_list)):
	augmentations = transform(image = image_list[i], mask = mask_list[i])
	augmented_img = augmentations["image"]
	augmented_mask = augmentations["mask"]
	Image.fromarray(augmented_img).save(target_dir + '/image' + str(i) + '.jpg')
	Image.fromarray(augmented_mask).save(target_masks + '/mask' + str(i) + '.jpg')
