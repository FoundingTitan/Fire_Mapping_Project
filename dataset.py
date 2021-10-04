import os
from PIL import Image
from torch.utils.data import Dataset

class FireDataset(Dataset) :
    
    def __init__(self, img_dir, mask_dir, transform_img, transform_mask, return_name=False) :
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = os.listdir(self.img_dir)
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.return_name = return_name
    
    def __len__(self) :
        return len(self.img_names)

    def __getitem__(self,idx) :
        
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        mask_name = img_name.split('.')[0] + '.jpg'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        if self.return_name:
            return img, mask, img_name.split('.')[0]
        
        return img, mask