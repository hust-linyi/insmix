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

class NucleiDataset(BaseDataset):
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

        self.imgs = np.load(os.path.join(opt.dataroot, opt.phase, 'data_after_stain_norm_ref1.npy'))
        self.gts = np.load(os.path.join(opt.dataroot, opt.phase, 'gt.npy'))
        self.A_size = self.imgs.shape[0]  # get the size of dataset A
        self.B_size = self.A_size
        self.crop_size = opt.crop_size

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        imgA = self.imgs[index].copy()
        maskA = self.gts[index].copy()
        index_B = random.randint(0, self.A_size - 1)
        imgB = self.imgs[index_B].copy()
        maskB = self.gts[index_B].copy()

        h, w, _ = imgA.shape
        sha = np.random.randint(0, h-self.crop_size-1)
        swa = np.random.randint(0, w-self.crop_size-1)
        imga = imgA[sha:sha+self.crop_size, swa:swa+self.crop_size, :].copy()
        maska = maskA[sha:sha+self.crop_size, swa:swa+self.crop_size].copy()

        shb = np.random.randint(0, h-self.crop_size-1)
        swb = np.random.randint(0, w-self.crop_size-1)
        imgb = imgB[shb:shb+self.crop_size, swb:swb+self.crop_size, :].copy()
        maskb = maskB[shb:shb+self.crop_size, swb:swb+self.crop_size].copy()

        imga, maska, maskb = simple_insmix(imga, imgb, maska, maskb)

        shc = np.random.randint(0, h-self.crop_size-1)
        swc = np.random.randint(0, w-self.crop_size-1)
        img_rand_real1 = imgA[shc:shc+self.crop_size, swc:swc+self.crop_size, :].copy()
        shc = np.random.randint(0, h-self.crop_size-1)
        swc = np.random.randint(0, w-self.crop_size-1)
        img_rand_real2 = imgA[shc:shc+self.crop_size, swc:swc+self.crop_size, :].copy()

        imga, maska, maskb = nuclei_transform(imga, maska, maskb)
        img_rand_real1, _, _ = nuclei_transform(img_rand_real1)
        img_rand_real2, _, _ = nuclei_transform(img_rand_real2)


        maska = np.concatenate((maska[np.newaxis, ...], maska[np.newaxis, ...], maska[np.newaxis, ...]), axis=0)
        maskb = np.concatenate((maskb[np.newaxis, ...], maskb[np.newaxis, ...], maskb[np.newaxis, ...]), axis=0)
        return {'A': imga.copy(), 'B': img_rand_real1.copy(), 'C': img_rand_real2.copy(), 'maska': maska.copy(), 'maskb': maskb.copy()}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


def nuclei_transform(img, mask1=None, mask2=None):
    if mask1 is None:
        mask1 = np.zeros((img.shape[1], img.shape[2]))
    if mask2 is None:
        mask2 = np.zeros((img.shape[1], img.shape[2]))
    h, w, c = img.shape
    img = np.asarray(img).astype(np.float32)
    img = img.transpose([2, 0, 1])
    for imod in list(range(c)):
        img[imod] = (img[imod]/255.0 - norm_mean[imod])/norm_std[imod]

    # flip
    if np.random.rand() > 0.5:
        img = img[:, :, ::-1]
        mask1 = mask1[:, ::-1]
        mask2 = mask2[:, ::-1]
    
    # rotation
    if np.random.rand() > 0.5:
        rot_fac = np.random.randint(1, 3)
        img = np.rot90(img, rot_fac, (len(img.shape)-2, len(img.shape)-1))
        mask1 = np.rot90(mask1, rot_fac, (len(mask1.shape)-2, len(mask1.shape)-1))
        mask2 = np.rot90(mask2, rot_fac, (len(mask2.shape)-2, len(mask2.shape)-1))
    return img, mask1, mask2
