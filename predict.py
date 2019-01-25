import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import argparse
import numpy as np
import pandas as pd
import scipy.io
import skimage.io
import sys
import tqdm

from PIL import Image, ImageFilter
from tensorboardX import SummaryWriter

from model.FCN8s import FCN8s
from model.SegNetBasic import SegNetBasic
from model.UNet import UNet
from dataset import PartAffordanceDataset, CenterCrop, ToTensor, Normalize
from dataset import crop_center_numpy, crop_center_pil_image


""" default """

MODEL = 'FCN8s'
PARAMS_PATH = None
NUM_IMAGES = 8
IN_CHANNEL = 3
N_CLASSES = 8   # Part Affordance Dataset has 8 classes including background
DEVICE = 'cpu'
RESULT_PATH = './' + MODEL + '_result/'



def get_arguments():
    """
    Parse all the arguments from the command line interface

    return a list of parse arguments
    """

    parser = argparse.ArgumentParser(description='train network for affordance detection')

    parser.add_argument("--model", type=str, default=MODEL,
                        help="available model options => FCN8s/SegNetBasic/UNet")
    parser.add_argument("--params_path", type=str, default=PARAMS_PATH,
                        help="if you want to use a trained model, input the path of a file of it")
    parser.add_argument("--num_images", type=int, default=NUM_IMAGES,
                        help="number of images to predict for segmentation")
    parser.add_argument("--in_channel", type=int, default=IN_CHANNEL,
                        help="the number of the channel of input images")
    parser.add_argument("--n_classes", type=int, default=N_CLASSES,
                        help="number of classes in the dataset including background")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="the device you'll use (cpu or cuda:0 or so on)")
    parser.add_argument("--result_path", type=str, default=RESULT_PATH,
                        help="select your directory to save the result")

    return parser.parse_args()


args = get_arguments()



""" visulalization """
# mean and std of each channel after centercrop
mean=[0.2191, 0.2349, 0.3598]
std=[0.1243, 0.1171, 0.0748]

def reverse_normalize(x, mean, std):
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x


# assign the colors to each class
colors = torch.tensor([[0, 0, 0],         # class 0 'background'  black
                        [255, 0, 0],       # class 1 'grasp'       red
                        [255, 255, 0],     # class 2 'cut'         yellow
                        [0, 255, 0],       # class 3 'scoop'       green
                        [0, 255, 255],     # class 4 'contain'     sky blue
                        [0, 0, 255],       # class 5 'pound'       blue
                        [255, 0, 255],     # class 6 'support'     purple
                        [255, 255, 255]    # class 7 'wrap grasp'  white
                        ])

# convert class prediction to the mask
def class_to_mask(cls):
    
    mask = colors[cls].transpose(1, 2).transpose(1, 3)
    
    return mask


def predict(model, sample, device='cpu'):
    model.eval()
    model.to(device)
    
    x, y = sample['image'], sample['class']
    
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        _, y_pred = model(x).max(1)    # y_pred.shape => (N, 240, 320)

    true_mask = class_to_mask(y).to('cpu')
    pred_mask = class_to_mask(y_pred).to('cpu')
    
    save_image(true_mask, args.result_path + '/' + 'true_masks_with_' + args.model + '.jpg')
    save_image(pred_mask, args.result_path + '/' + 'predicted_masks_with_' + args.model + '.jpg')



if __name__ == '__main__':
    
    if args.model == 'FCN8s':
        model = FCN8s(args.in_channel, args.n_classes)
        
        # for FCN8s, input size is (256, 320). for others, input size is (240, 320)
        # this is because　input size for FCN8s, which has VGG architecture,
        # must be a multiple of 32
        class CenterCrop(object):
            def __call__(self, sample):
                image, cls = sample['image'], sample['class']
                image = Image.fromarray(np.uint8(image))
        
                image = crop_center_pil_image(image, 320, 256)
                cls = crop_center_numpy(cls, 256, 320)   
                image = np.asarray(image)
        
                return {'image': image, 'class': cls}
        
    elif args.model == 'SegNetBasic':
        model = SegNetBasic(args.in_channel, args.n_classes)
    elif args.model == 'UNet':
        model = UNet(args.in_channel, args.n_classes)
    else:
        print('This model doesn\'t exist in the model directory')
        sys.exit(1)

    if args.params_path is not None:
        model.load_state_dict(torch.load(args.params_path, 
                                        map_location=lambda storage, loc: storage))


    """ define DataLoader """

    data = PartAffordanceDataset('test.csv',
                            transform=transforms.Compose([
                                CenterCrop(),
                                ToTensor(),
                                Normalize()
                            ]))

    data_loader = DataLoader(data, batch_size=args.num_images, shuffle=True)



    for sample in data_loader:
        model.eval()
        
        predict(model, sample, device=args.device)

        x = sample["image"]
        x = reverse_normalize(x, mean, std)
        save_image(x, args.result_path + '/' + 'original_images_with_' + args.model + '.jpg')
        
        break






