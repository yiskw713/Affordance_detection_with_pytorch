import glob
import pandas as pd

"""
use only rgb images and class labels in Part Affordance Dataset
please visit this website:
http://users.umiacs.umd.edu/~amyers/part-affordance-dataset/

"""

path_list = glob.glob('part-affordance-dataset/tools/*')


""" pick up 50 images per folder for validation and the rest is for train """

image_path_list_train = []
image_path_list_test = []

for path in path_list:
    img_path = glob.glob(path + '/*rgb.jpg')
    image_path_list_test += img_path[:50]
    image_path_list_train += img_path[50:]


class_path_list_train = []
class_path_list_test = []

for path in image_path_list_train:
    cls_path = path[:-7] + 'label.mat'
    class_path_list_train.append(cls_path)

for path in image_path_list_test:
    cls_path = path[:-7] + 'label.mat'
    class_path_list_test.append(cls_path)


''' save the path list as csv file.'''

df_train = pd.DataFrame({
    'image_path': image_path_list_train,
    'class_path': class_path_list_train},
    columns=['image_path', 'class_path']
)

df_test = pd.DataFrame({
    "image_path": image_path_list_test,
    "class_path": class_path_list_test},
    columns=["image_path", "class_path"]
)

all_data = pd.concat([df_train, df_test])

df_train.to_csv('train.csv', index=None)
df_test.to_csv('test.csv', index=None)
all_data.to_csv('all_data.csv', index=None)