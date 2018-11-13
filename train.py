import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
CLASS_WEIGHT = True
BATCH_SIZE = 10
NUM_WORKER = 4
MAX_EPOCH = 200
LEARNING_RATE = 0.001
IN_CHANNEL = 3
N_CLASSES = 8   # Part Affordance Dataset has 8 classes including background
DEVICE = 'cpu'
WRITER = True
RESULT_PATH = './' + MODEL + '_result/'



def get_arguments():
    """
    Parse all the arguments from the command line interface

    return a list of parse arguments
    """

    parser = argparse.ArgumentParser(description='train network for affordance detection')

    parser.add_argument("--model", type=str, default=MODEL,
                        help="available model options => FCN8s/SegNetBasic/UNet")
    parser.add_argument("--class_weight", type=bool, default=CLASS_WEIGHT,
                        help="if you want to use class weight, input True. Else, input False")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="number of batch size: number of samples sent to the network at a time")
    parser.add_argument("--num_worker", type=int, default=NUM_WORKER,
                        help="number of workers for multithread data loading")
    parser.add_argument("--max_epoch", type=int, default=MAX_EPOCH,
                        help="the number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="base learning rate for training")
    parser.add_argument("--in_channel", type=int, default=IN_CHANNEL,
                        help="the number of the channel of input images")
    parser.add_argument("--n_classes", type=int, default=N_CLASSES,
                        help="number of classes in the dataset including background")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="the device you'll use (cpu or cuda:0 or so on)")
    parser.add_argument("--writer", type=bool, default=WRITER,
                        help="if you want to use SummaryWriter in tesorboardx, input True. Else, input False")
    parser.add_argument("--result_path", type=str, default=RESULT_PATH,
                        help="select your directory to save the result")

    return parser.parse_args()


args = get_arguments()


""" class weight after center crop """

if args.class_weight:
    class_num = torch.tensor([2078085712, 34078992, 15921090, 12433420, 
                             38473752, 6773528, 9273826, 20102080])

    total = class_num.sum().item()

    frequency = class_num.float() / total
    median = torch.median(frequency)

    class_weight = median / frequency


""" if you try to count the number of pixels in each class, 
    please try this code:

data = PartAffordanceDataset('all_data.csv',
                                transform=transforms.Compose([
                                    CenterCrop(),
                                    ToTensor(),
                                    Normalize()
                                ]))

data_loader = DataLoader(data, batch_size=100, shuffle=False)

cnt_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}

for sample in data_laoder:
    img = sample['class'].numpy()
    
    num, cnt = np.unique(img, return_counts=True)    
    for n, c in zip(num, cnt):
        cnt_dict[n] += c

"""


""" validation """

def eval_model(model, test_loader, device='cpu'):
    model.eval()
    
    intersection = torch.zeros(8)   # the dataset has 8 classes including background
    union = torch.zeros(8)
    
    for sample in test_loader:
        x, y = sample['image'], sample['class']
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            _, ypred = model(x).max(1)    # y_pred.shape => (N, 240, 320)
        
        for i in range(8):
            y_i = (y == i)           
            ypred_i = (ypred == i)   

            inter = (y_i.byte() & ypred_i.byte()).float().sum().to('cpu')
            intersection[i] += inter
            union[i] += (y_i.float().sum() + ypred_i.float().sum()).to('cpu') - inter
    
    """ iou[i] is the IoU of class i """
    iou = intersection / union
    
    return iou



""" initialize weight """

def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



def main():

    """ DataLoader """
    train_data = PartAffordanceDataset('train.csv',
                                    transform=transforms.Compose([
                                        CenterCrop(),
                                        ToTensor(),
                                        Normalize()
                                    ]))

    test_data = PartAffordanceDataset('test.csv',
                                    transform=transforms.Compose([
                                        CenterCrop(),
                                        ToTensor(),
                                        Normalize()
                                    ]))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)


    if args.model == 'FCN8s':
        model = FCN8s(args.in_channel, args.n_classes)
    elif args.model == 'SegNetBasic':
        model = SegNetBasic(args.in_channel, args.n_classes)
    elif args.model == 'UNet':
        model = UNet(args.in_channel, args.n_classes)
    else:
        print('This model doesn\'t exist in the model directory')
        sys.exit(1)

    model.apply(init_weight)



    """ training """

    if args.writer:
        writer = SummaryWriter(args.result_path)

    if args.class_weight:
        criterion = nn.CrossEntropyLoss(weight=class_weight.to(args.device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    train_losses = []
    val_iou = []
    mean_iou = []
    best_mean_iou = 0.0


    for epoch in range(args.max_epoch):
        model.train()
        running_loss = 0.0
        
        for i, sample in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            
            x, y = sample['image'], sample['class']
            
            x = x.to(args.device)
            y = y.to(args.device)

            h = model(x)
            loss = criterion(h, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        train_losses.append(running_loss / i)
        
        val_iou.append(eval_model(model, test_loader, args.device).to('cpu').float())
        mean_iou.append(val_iou[-1].mean().item())
        
        if best_mean_iou < mean_iou[-1]:
            best_mean_iou = mean_iou[-1]
            torch.save(model.state_dict(), args.result_path + '/best_mean_iou_model.prm')
        
        if writer is not None:
            writer.add_scalar("train_loss", train_losses[-1], epoch)
            writer.add_scalar("mean_IoU", mean_iou[-1], epoch)
            writer.add_scalars("class_IoU", {'iou of class 0': val_iou[-1][0],
                                           'iou of class 1': val_iou[-1][1],
                                           'iou of class 2': val_iou[-1][2],
                                           'iou of class 3': val_iou[-1][3],
                                           'iou of class 4': val_iou[-1][4],
                                           'iou of class 5': val_iou[-1][5],
                                           'iou of class 6': val_iou[-1][6],
                                           'iou of class 7': val_iou[-1][7]}, epoch)
            
        print('epoch: {}\tloss: {:.5f}\tmean IOU: {:.3f}'.format(epoch, train_losses[-1], mean_iou[-1]))
        
    torch.save(model.state_dict(), args.result_path +"/final_model.prm")


if __name__ == '__main__':
    main()
