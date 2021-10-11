import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random

from torchvision import transforms
import torchvision.transforms.functional as TF

class FireDataset(Dataset) :
    
    def __init__(self, img_dir, mask_dir, transform_mode, return_name=False) :
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = os.listdir(self.img_dir)
        self.return_name = return_name

        self.transform_basic = transforms.Compose([
        transforms.Resize([256,256]), #Resize the input image to this size
        transforms.ToTensor()])
        self.transform_mode = transform_mode
    
    def __len__(self) :
        return len(self.img_names)

    def transform(self, image, mask):
        # Resize

        sizes = [256, 280, 300]
        random_size =  np.random.choice(sizes, 1)[0]
        # print(random_size)
        resize = transforms.Resize(size=(random_size, random_size))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(256, 256))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __getitem__(self,idx) :
        
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        mask_name = img_name.split('.')[0] + '.jpg'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform_mode == "basic":
            img = self.transform_basic(img)
            mask = self.transform_basic(mask)
        elif self.transform_mode == "crop_hflip_vflip":
            img, mask = self.transform(img, mask)

        if self.return_name:
            return img, mask, img_name.split('.')[0]
        
        return img, mask



