import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html, util
import numpy as np
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    # times, num_img, h, w, c = dataset.imgs.shape
    out_img = np.zeros_like(dataset.dataset.imgs)
    print(len(dataset))
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        # img_path = model.get_image_paths()     # get image paths
        img_path = './imgs/'
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        for label, im_data in visuals.items():
            if label == 'fake_B':
                continue
            t, idx = i % dataset.dataset.imgs.shape[0], i // dataset.dataset.imgs.shape[0]
            im = util.tensor2im(im_data)
            util.save_image(im, os.path.join(img_path, '%04d_%04d_%s.png' % (t, idx, label)))
            out_img[t, idx, :, :, :] = im
        np.save(opt.save_dir, out_img)
    webpage.save()  # save the HTML

