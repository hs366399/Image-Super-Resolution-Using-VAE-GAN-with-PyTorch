#importing required libraries
import os
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

print(torch.cuda.get_device_name(0))

class ImageDataset(Dataset):
    
    def __init__(self, data_path, transform = None):
        self.train_data_path = data_path['data']
        self.train_label_path = data_path['labels']
        self.train_lables = os.listdir(self.train_label_path)
        self.train_data = os.listdir(self.train_data_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, indx):
        
        if indx >= len(self.train_data):
            raise Exception("Index should be less than {}".format(len(self.train_data)))
        image = Image.open(self.train_data_path + self.train_data[indx]).convert('RGB')
        final_label = cv2.resize(cv2.imread(self.train_label_path + self.train_lables[indx]), (480, 480))
        final_label = Image.fromarray(cv2.cvtColor(final_label, cv2.COLOR_BGR2RGB))
        
        if len(self.transform) > 0:
            image = self.transform(image)
            final_label = self.transform(final_label)

        return image, final_label
