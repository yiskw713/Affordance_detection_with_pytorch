# Affordance detection with pytorch
this repository is the implementation to detect affordances of objects with pytorch

# requirements
* python 3
* pytorch >= 0.4
* tensorboardx
* scipy, skimage, tqdm


# dataset
use the following dataset:

[Part Affordance Dataset](http://users.umiacs.umd.edu/~amyers/part-affordance-dataset/)

Affordance Detection of Tool Parts from Geometric Features,  
Austin Myers, Ching L. Teo, Cornelia Fermüller, Yiannis Aloimonos.  
International Conference on Robotics and Automation (ICRA). 2015.  

# training on Part Affordance Dataset
please `run train.py` in the command line after downloading the dataset in 'part-affordance-dataset' directory.

```
python train.py -h
```

Then,

```
usage: train.py [-h] [--model MODEL] [--class_weight CLASS_WEIGHT]
                [--batch_size BATCH_SIZE] [--num_worker NUM_WORKER]
                [--max_epoch MAX_EPOCH] [--learning_rate LEARNING_RATE]
                [--in_channel IN_CHANNEL] [--n_classes N_CLASSES]
                [--device DEVICE] [--writer WRITER]
                [--result_path RESULT_PATH]

train network for affordance detection

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         available model options => FCN8s/SegNetBasic/UNet
  --class_weight CLASS_WEIGHT
                        if you want to use class weight, input True. Else,
                        input False
  --batch_size BATCH_SIZE
                        number of batch size: number of samples sent to the
                        network at a time
  --num_worker NUM_WORKER
                        number of workers for multithread data loading
  --max_epoch MAX_EPOCH
                        the number of epochs for training
  --learning_rate LEARNING_RATE
                        base learning rate for training
  --in_channel IN_CHANNEL
                        the number of the channel of input images
  --n_classes N_CLASSES
                        number of classes in the dataset including background
  --device DEVICE       the device you'll use (cpu or cuda:0 or so on)
  --writer WRITER       if you want to use SummaryWriter in tesorboardx, input
                        True. Else, input False
  --result_path RESULT_PATH
                        select your directory to save the result
```
you can choose the model, batch size and so on as you like


# predict
To predict affordance of images in the dataset, run `python predict.py` in the command line.  
Just like trainig, you can select the parameter, path and so on.

```
python predict.py -h

usage: predict.py [-h] [--model MODEL] [--params_path PARAMS_PATH]
                  [--num_images NUM_IMAGES] [--in_channel IN_CHANNEL]
                  [--n_classes N_CLASSES] [--device DEVICE]
                  [--result_path RESULT_PATH]

train network for affordance detection

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         available model options => FCN8s/SegNetBasic/UNet
  --params_path PARAMS_PATH
                        if you want to use a trained model, input the path of
                        a file of it
  --num_images NUM_IMAGES
                        number of images to predict for segmentation
  --in_channel IN_CHANNEL
                        the number of the channel of input images
  --n_classes N_CLASSES
                        number of classes in the dataset including background
  --device DEVICE       the device you'll use (cpu or cuda:0 or so on)
  --result_path RESULT_PATH
                        select your directory to save the result
```


# date
Oct. 26, 2018
