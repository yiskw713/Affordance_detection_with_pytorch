import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd
import numpy as np
import scipy.io
import skimage.io

from PIL import Image, ImageFilter

"""
Define the dataset class for Part Affordance Dataset
pre-processing is done as below.

center crop => conver to torch.tensor => normalization

"""


class PartAffordanceDataset(Dataset):
    """Part Affordance Dataset"""
    
    def __init__(self, csv_file, transform=None):
        super().__init__()
        
        self.image_class_path = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_class_path)
    
    def __getitem__(self, idx):
        image_path = self.image_class_path.iloc[idx, 0]
        class_path = self.image_class_path.iloc[idx, 1]
        image = skimage.io.imread(image_path) # read as numpy array
        cls = scipy.io.loadmat(class_path)["gt_label"]
        
        sample = {'image': image, 'class': cls}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample



""" transform for pre-processing """


""" crop images and labels at the center """

def crop_center_numpy(array, crop_height, crop_weight):
    h, w = array.shape
    return array[h//2 - crop_height//2: h//2 + crop_height//2,
                 w//2 - crop_weight//2: w//2 + crop_weight//2
                ]

def crop_center_pil_image(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

class CenterCrop(object):
    def __call__(self, sample):
        image, cls = sample['image'], sample['class']
        
        image = Image.fromarray(np.uint8(image))
        
        image = crop_center_pil_image(image, 320, 240)
        cls = crop_center_numpy(cls, 240, 320)
        
        image = np.asarray(image)
        
        return {'image': image, 'class': cls}



""" convert both images and labels to torch.tensor """

class ToTensor(object):
    def __call__(self, sample):
        image, cls = sample['image'], sample['class']
        
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(), 
                'class': torch.from_numpy(cls).long()}


""" normalize images """
class Normalize(object):
    def __init__(self):
        # mean and std of each channel after centercrop
        self.mean=[55.8630, 59.9099, 91.7419]
        self.std=[31.6852, 29.8496, 19.0835]
    
    def __call__(self, sample):
        image, cls = sample['image'], sample['class']
        
        image = transforms.functional.normalize(image, self.mean, self.std)
        
        return {'image': image, 'class': cls}


"""
if you want to calculate mean and std by yourself, try this code:


data = PartAffordanceDataset('image_class_path.csv',
                                transform=transforms.Compose([
                                    CenterCrop(),
                                    ToTensor()
                                ]))

data_loader = DataLoader(data, batch_size=10, shuffle=False)

mean = 0
std = 0
n = 0

for sample in data_loader:
    img = sample['image']   
    img = img.view(len(img), 3, -1)
    mean += img.mean(2).sum(0)
    std += img.std(2).sum(0)
    n += len(img)
    
mean /= n
std /= n

"""