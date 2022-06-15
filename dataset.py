from email.contentmanager import raw_data_manager
from genericpath import exists
import torch
import torch.utils.data as data
import os
from PIL import Image, ImageEnhance
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter, median_filter
import numpy as np
import pandas as pd
import random
from glob import glob
import mix 
import math
from data_aug import insmix, bg_shuffle, elastic_transform


norm_mean=[0.485, 0.456, 0.406]
norm_std=[0.229, 0.224, 0.225]

def data_split(x, y, y_bnd, y_cent, fold=0, phase='train'):
    """
    x: (c, h, w)
    y: (h, w)
    y_bnd: (h, w)
    """
    validnum = int(x.shape[0] * 0.2)
    valstart = fold * validnum
    valend = (fold + 1) * validnum
    if phase == 'train':
        x = np.concatenate([x[:valstart], x[valend:]], axis=0)
        y = np.concatenate([y[:valstart], y[valend:]], axis=0)
        y_bnd = np.concatenate([y_bnd[:valstart], y_bnd[valend:]], axis=0)
        y_cent = np.concatenate([y_cent[:valstart], y_cent[valend:]], axis=0)
    else:
        x = np.concatenate(x[valstart:valend])
        y = np.concatenate(y[valstart:valend])
        y_bnd = np.concatenate(y_bnd[valstart:valend])
        y_cent = np.concatenate(y_cent[valstart:valend])
    return x, y, y_bnd, y_cent


class DataFolder(data.Dataset):
    def __init__(self, root_dir, phase, fold, data_transform=None):
        """
        :param root_dir: 
        :param data_transform: data transformations
        """
        super(DataFolder, self).__init__()
        self.data_transform = data_transform
        self.phase = phase
        self.imgs = np.load(os.path.join(root_dir, 'data_after_stain_norm_ref1.npy'))
        self.bnd_labels = np.load(os.path.join(root_dir,  'bnd.npy'))
        self.seg_labels = np.load(os.path.join(root_dir,  'ist.npy'))
        self.cent_labels = np.load(os.path.join(root_dir, 'cent.npy'))
        self.imgs, self.seg_labels, self.bnd_labels, self.cent_labels = data_split(self.imgs, self.seg_labels, self.bnd_labels, self.cent_labels, fold, self.phase) 

        
        self.nimg = self.imgs.shape[0]
        self.naug = 8 # ori*1 + rotate*3 + flip*2 + insmix*2
        self.crop_size = [256, 256]


    def __len__(self):
        return self.nimg*self.naug


    def __getitem__(self, idx):
        iaug = int(np.mod(idx, self.naug))
        index = int(np.floor(idx/self.naug))

        img = self.imgs[index].copy()
        seg_label = self.seg_labels[index].copy()
        bnd_label = self.bnd_labels[index].copy()

        if self.phase == 'train':
            # data augmentation
            # While doing random augmentation here (instead of calling transformer) with multi-workers, all the workers get the same numpy random state sometimes. To avoid this, call np.random.seed() again here.
            np.random.seed()
            h, w, mod = img.shape

            # Color, Brightness, Contrast, Sharpness
            rnd_factor = np.random.rand()*0.1+0.9
            img = Image.fromarray(img.astype(np.uint8))
            img = ImageEnhance.Color(img).enhance(rnd_factor)
            rnd_factor = np.random.rand()*0.1+0.9
            img = ImageEnhance.Brightness(img).enhance(rnd_factor)
            rnd_factor = np.random.rand()*0.1+0.9
            img = ImageEnhance.Contrast(img).enhance(rnd_factor)
            rnd_factor = np.random.rand()*0.2+0.9
            img = ImageEnhance.Sharpness(img).enhance(rnd_factor)
            img = np.asarray(img).astype(np.float32)
            img = img.transpose([2, 0, 1])
            for imod in list(range(mod)):
                img[imod] = (img[imod]/255.0 - norm_mean[imod])/norm_std[imod]
            img += np.random.normal(0, np.random.rand(), img.shape)*0.01

            # Crop
            sh = np.random.randint(0, h-self.crop_size[0]-1)
            sw = np.random.randint(0, w-self.crop_size[1]-1)
            if self.phase == 'train' and iaug > 6:
                cent_label = self.cent_labels[index].copy()
                img, seg_label, bnd_label = insmix(img, cent_label, seg_label, bnd_label, sh, sw, self.crop_size[0])
                img = bg_shuffle(img, seg_label, anchor_size=20, bg_shift_radio=0.5)
            else:
                img = img[:, sh:(sh+self.crop_size[0]), sw:(sw+self.crop_size[1])]
                seg_label = seg_label[sh:(sh+self.crop_size[0]), sw:(sw+self.crop_size[1])] > 0
                bnd_label = bnd_label[sh:(sh+self.crop_size[0]), sw:(sw+self.crop_size[1])] > 0

            # assert(iaug>0 and iaug<6)
            # Aug
            if iaug<=3 and iaug>0:
                img = np.rot90(img, iaug, axes=(len(img.shape)-2, len(img.shape)-1))
                seg_label = np.rot90(seg_label, iaug, axes=(len(seg_label.shape)-2, len(seg_label.shape)-1))
                bnd_label = np.rot90(bnd_label, iaug, axes=(len(bnd_label.shape)-2, len(bnd_label.shape)-1))
            elif iaug>=4 and iaug<6:
                img = np.flip(img, len(img.shape)-(iaug-3))
                seg_label = np.flip(seg_label, len(seg_label.shape)-(iaug-3))
                bnd_label = np.flip(bnd_label, len(bnd_label.shape)-(iaug-3))

            if np.random.rand()>=0.5:
                rnd_et = np.random.rand(2)
                indices = elastic_transform(seg_label.shape, int(rnd_et[0]*20), 5*(rnd_et[1]+1.0))
                for imod in range(mod):
                    img[imod] = map_coordinates(img[imod].squeeze(), indices, order=1, mode='reflect').reshape(img[imod].shape)
                seg_label = map_coordinates(seg_label.squeeze(), indices, order=1, mode='reflect').reshape(seg_label.shape)
                bnd_label = map_coordinates(bnd_label.squeeze(), indices, order=1, mode='reflect').reshape(bnd_label.shape)

        img = img.copy()
        seg_label = seg_label.copy()
        bnd_label = bnd_label.copy()
        
        return img, seg_label, bnd_label 

