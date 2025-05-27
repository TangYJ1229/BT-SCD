import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import datasets.transform as transform
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import cv2

# SECOND
ST_COLORMAP = [[255,255,255], [0,128,0], [128,128,128], [0,255,0], [0,0,255], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'low vegetation', 'ground', 'tree', 'water', 'building', 'sports field']

# JL1H
# ST_COLORMAP = [[255,255,255], [0,128,0], [0,0,128], [128,0,0], [0,128,128], [128,128,0]]
# ST_CLASSES = ['unchanged', 'farm', 'road', 'tree', 'building', 'other']

# Landsat
# ST_COLORMAP = [[255,255,255], [0,155,0], [255,165,0], [230,30,100], [0,170,240]]
# ST_CLASSES = ['unchanged', 'farmland', 'desert', 'building', 'water']

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B  = np.array([49.41,  47.01,  47.94])

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def boundary2Color(pred):
    colormap = np.asarray([[0,0,0], [255,255,255]], dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time == 'A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im

def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs


class Data(data.Dataset):
    def __init__(self, datapath, mode, augmentation=False):
        self.datapath = datapath
        self.mode = mode
        self.augmentation = augmentation

        self.A = os.path.join(datapath, "A")
        self.B = os.path.join(datapath, "B")
        self.labels_A = os.path.join(datapath, "label1")
        self.labels_B = os.path.join(datapath, "label2")

        self.list_img = self.get_mask_name(datapath)
    
    def get_mask_name(self, datapath):
        images_list_file = os.path.join(datapath, 'list', self.mode + ".txt")
        with open(images_list_file, "r") as f:
            return f.readlines()


    def __getitem__(self, idx):
        imgname = self.list_img[idx].strip('\n')

        img_A = io.imread(os.path.join(self.A, imgname + '.png'))
        img_B = io.imread(os.path.join(self.B, imgname + '.png'))
        label_A = io.imread(os.path.join(self.labels_A, imgname + '.png'))
        label_B = io.imread(os.path.join(self.labels_B, imgname + '.png'))
        # img_A = io.imread(os.path.join(self.A, imgname))
        # img_B = io.imread(os.path.join(self.B, imgname))
        # label_A = io.imread(os.path.join(self.labels_A, imgname))
        # label_B = io.imread(os.path.join(self.labels_B, imgname))

        if self.augmentation:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_MCD(img_A, img_B, label_A, label_B)
            # img_A, img_B, label_A, label_B = transform.RandomExchange(img_A, img_B, label_A, label_B)

        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')


        label_bn = np.zeros_like(label_A)
        label_bn[label_A != 0] = 1
        b_label_bn = cv2.Canny(label_bn, 0, 0) // 255

        b_label_sem = np.zeros_like(label_A)
        b_label_A = cv2.Canny(label_A, 0, 0) // 255
        b_label_B = cv2.Canny(label_B, 0, 0) // 255
        b_label_sem[(b_label_A + b_label_B) != 0] = 1


        return F.to_tensor(img_A), F.to_tensor(img_B), \
               torch.from_numpy(label_A), torch.from_numpy(label_B), \
               torch.from_numpy(b_label_sem), torch.from_numpy(b_label_bn), imgname

    def __len__(self):
        return len(self.list_img)


