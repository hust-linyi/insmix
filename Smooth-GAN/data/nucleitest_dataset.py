import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
from PIL import Image, ImageEnhance


# Image Net mean and std
# norm_mean=[0.485, 0.456, 0.406]
# norm_std=[0.229, 0.224, 0.225]
norm_mean = [0.5, 0.5, 0.5]
norm_std = [0.5, 0.5, 0.5]
# norm_mean = [0.0, 0.0, 0.0]
# norm_std = [1., 1., 1.]


def simple_insmix(imga, imgb, maska, maskb):
    """
    apply translation, rotation, scaling, filp to maskb
    """
    maska[maskb > 0] = 0
    imgout = imga.copy()
    imgout[maskb > 0] = imgb[maskb > 0]
    # random translation
    # random rotation
    # random scaling
    # random flip

    return imgout, maska, maskb

class NucleiTestDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired nuclei datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.imgs = np.load(os.path.join('/home/ylindq/Data/kumar/np', 'gan', 'data_after_stain_norm_ref1.npy'))
        self.gts = np.load(os.path.join('/home/ylindq/Data/kumar/np', 'gan', 'gt.npy'))
        self.gts_add = np.load(os.path.join('/home/ylindq/Data/kumar/np', 'gan', 'gt_add.npy'))
        self.times, self.num_img, self.h, self.w, self.c = self.imgs.shape
        self.A_size = self.imgs.shape[0]  # get the size of dataset A
        self.B_size = self.A_size
        self.crop_size = 256


    def __getitem__(self, index):
        t, idx = index % self.times, index // self.times
        imga = self.imgs[t, idx, :, :, :].copy()
        maskb = self.gts_add[t, idx, :, :].copy()
        maska = self.gts[t, idx, :, :].copy()

        maska[maskb > 0] = 0
        imga = nuclei_transform(imga)

        img_rand_real1 = np.zeros_like(imga)
        img_rand_real2 = np.zeros_like(imga)

        maska = np.concatenate((maska[np.newaxis, ...], maska[np.newaxis, ...], maska[np.newaxis, ...]), axis=0)
        maskb = np.concatenate((maskb[np.newaxis, ...], maskb[np.newaxis, ...], maskb[np.newaxis, ...]), axis=0)
        return {'A': imga.copy(), 'B': img_rand_real1.copy(), 'C': img_rand_real2.copy(), 'maska': maska.copy(), 'maskb': maskb.copy()}

   
    def __len__(self):
        return self.times * self.num_img


def nuclei_transform(img):
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    h, w, c = img.shape
    img = np.asarray(img).astype(np.float32)
    img = img.transpose([2, 0, 1])
    for imod in list(range(c)):
        img[imod] = (img[imod]/255.0 - norm_mean[imod])/norm_std[imod]
    return img