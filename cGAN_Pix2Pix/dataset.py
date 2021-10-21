import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir, image_dir='x', label_dir='y'):
        self.root_dir = root_dir
        self.root_dir_x = os.path.join(root_dir, image_dir)
        self.root_dir_y = os.path.join(root_dir, label_dir)
        self.list_files_x = os.listdir(self.root_dir_x)
        self.list_files_y = os.listdir(self.root_dir_y)
        if len(self.list_files_x) != len(self.list_files_y):
            raise ValueError(f'Length of files {str(len(self.list_files_x))} list in input (x) \
            set mismatch length of labelled files {str(len(self.list_files_y))}')
        #self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files_x)

    def __getitem__(self, index):

        img_file_x = self.list_files_x[index]
        img_file_y = self.list_files_y[index]

        img_path_x = os.path.join(self.root_dir_x, img_file_x)
        img_path_y = os.path.join(self.root_dir_y, img_file_y)

        # (from 'CMYK' to 'RGB')
        image_x = np.array(Image.open(img_path_x).convert('RGB'))
        image_y = np.array(Image.open(img_path_y).convert('RGB'))

        # (If the both image and target are in the same image)
        #input_image = image[:, :600, :]
        #target_image = image[:, 600:, :]

        # (Use for color images)
        input_image = image_y
        target_image = image_x

        # (Use for bw images)
        # input_image = np.stack((image_y,)*3, axis=-1)
        # target_image = np.stack((image_x,)*3, axis=-1)

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
