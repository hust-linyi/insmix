# InsMix

<!-- [[paper](https://arxiv.org/abs/1905.06696).] -->

This is the official code for "InsMix: Towards Realistic Generative Data Augmentation for Nuclei Instance Segmentation (MICCAI 2022, early accepted)"

### Pipeline
![pipeline](figs/insmix1.png)

### Method
![method](figs/insmix2.png)

### Usage
#### InsMix w/o Smooth-GAN
The fuctions 'insmix' and 'background shuffle' can be found in 'data_aug.py'.
The example code for dataloader is in 'dataset.py'. Note that it can be used to [BRPNet](https://github.com/csccsccsccsc/brpnet) and [NB-Net](https://github.com/easycui/nuclei_segmentation), which utilize two types of label, i.e., the inner area and the boundary.
#### InsMix w/ Smooth-GAN
You may simply run the scripts as:
```
bash Smooth-GAN/scripts/train_nuclei.sh
bash Smooth-GAN/scripts/test_nuclei.sh
```

### Citation
Pleae cite the paper if you use the code.
```
@inproceedings{lin2022insmix,
  title={{InsMix}: Towards Realistic Generative Data Augmentation for Nuclei Instance Segmentation},
  author={Lin, Yi and Wang, Zeyu and Cheng, Kwang-Ting and Chen, Hao},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2022},
  organization={Springer}
}
```
### TODO
- [ ] Training and testing on Kumar dataset.
- [ ] Refactor the code to make it more readable.

### Acknowledgment 
The code of Smooth-GAN is heavily build on [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), thanks for their amazing work!
