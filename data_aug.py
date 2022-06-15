import numpy as np
import random
from genericpath import exists
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import random


def shape_scale_constraints(mask_a, mask_b):
    # shape and scale constraints
    shape_factor = np.sum(np.abs(mask_a - mask_b)) / max(np.sum(mask_a), np.sum(mask_b))
    scale_factor = min(np.sum(mask_a), np.sum(mask_b)) / max(np.sum(mask_a), np.sum(mask_b))
    return shape_factor, scale_factor


def insmix(x_all, cent_sort, y_ist_sort, y_bnd_sort, sh, sw, crop_size):
    """
    1. instance library, add 0 to 600
    2. distance aware
    3. shape constraints
    """
    x_patch = x_all[:, sh:sh+crop_size, sw:sw+crop_size]
    y_ist_patch = y_ist_sort[sh:sh+crop_size, sw:sw+crop_size] 
    y_gt_patch = y_ist_patch > 0
    y_bnd_patch = y_bnd_sort[sh:sh+crop_size, sw:sw+crop_size] > 0
    cent_patch = cent_sort[sh:sh+crop_size, sw:sw+crop_size]

    max_add = 0.5 * len(list(cent_patch[cent_patch > 0]))
    dis = np.random.randint(-40, 40)
    idx_add_list = list(cent_sort[cent_sort > 0])
    random.shuffle(idx_add_list)
    if min(max_add, len(idx_add_list)) < 2:
        num_add = 0
    else:
        num_add = np.random.randint(1, min(max_add, len(idx_add_list)))

    cent_a_id_list = list(cent_patch[cent_patch > 0].astype(int))
    cent_a_id_list.sort()
    for i in range(num_add):
        idx_add = idx_add_list[i]
        mask_ist = np.array(np.where(y_ist_sort==idx_add))
        mask_bnd = np.array(np.where(y_bnd_sort==idx_add))
        mask_cent = np.array(np.where(cent_sort==idx_add))

        # which nuclear to be augmented
        nuc_id = cent_a_id_list[i % len(cent_a_id_list)]
        center_point = np.where(cent_patch == nuc_id)

        shift_h = mask_cent[0][0] - center_point[0][0]
        shift_w = mask_cent[1][0] - center_point[1][0]

        mask_ist_a = mask_ist.copy()
        mask_ist_a[0] = mask_ist_a[0] - shift_h + dis
        mask_ist_a[1] = mask_ist_a[1] - shift_w + dis

        mask_bnd_a = mask_bnd.copy()
        mask_bnd_a[0] = mask_bnd_a[0] - shift_h + dis
        mask_bnd_a[1] = mask_bnd_a[1] - shift_w + dis

        # remove out of bound
        pos_ist = np.all(mask_ist_a >= 0, axis=0) * np.all(mask_ist_a < crop_size, axis=0)
        pos_bnd = np.all(mask_bnd_a >= 0, axis=0) * np.all(mask_bnd_a < crop_size, axis=0)
        mask_ist = mask_ist[:, pos_ist]
        mask_ist_a = mask_ist_a[:, pos_ist]
        mask_bnd = mask_bnd[:, pos_bnd]
        mask_bnd_a = mask_bnd_a[:, pos_bnd]
        mask_ist, mask_ist_a, mask_bnd, mask_bnd_a = map(tuple, [mask_ist, mask_ist_a, mask_bnd, mask_bnd_a])

        # shape constraints
        shape_constraints = 0.8
        scale_constraints = 0.5
        mask_a, mask_b = np.zeros_like(y_ist_patch), np.zeros_like(y_ist_patch)
        mask_b_idx = np.array(mask_ist_a) 
        mask_b_idx = np.clip(mask_b_idx - dis, 0, crop_size-1)
        mask_a[y_ist_patch==nuc_id] = 1
        mask_b_idx = tuple(mask_b_idx)
        mask_b[mask_b_idx] = 1
        shape_factor, scale_factor = shape_scale_constraints(mask_a, mask_b)
        # print(shape_factor, scale_factor)
        if shape_factor > shape_constraints or scale_factor < scale_constraints:
            continue
        
        # mix
        for i in range(x_all.shape[0]):
            x_patch[i][mask_ist_a] = x_all[i][mask_ist]
        y_gt_patch[mask_ist_a] = y_ist_sort[mask_ist]
        y_bnd_patch[mask_ist_a] = 0
        y_bnd_patch[mask_bnd_a] = y_bnd_sort[mask_bnd]
    return x_patch, y_gt_patch, y_bnd_patch


def pashuffle(num, perc=0.5):
    """
    num: number of patches
    perc: percentage of patches to be shuffled
    """
    num_shuffle = int(num * perc)
    idx_shuffle = np.random.choice(num, num_shuffle, replace=False)
    return idx_shuffle

def bg_shuffle(img, label, anchor_size, bg_shift_radio=0.5):
    """
    background shift
    img_shape: (c, h, w)
    """    
    img_out = img.copy()
    x = np.arange(0, img.shape[1] - anchor_size, anchor_size)
    y = np.arange(0, img.shape[2] - anchor_size, anchor_size)

    xx, yy = np.meshgrid(x, y)
    anchor_center = np.stack([xx, yy], axis=2).reshape(-1, 2)
    boxes = np.concatenate([anchor_center, anchor_center + anchor_size], axis=1)
    
    # remove the foreground patches
    keep = np.ones_like(boxes, dtype=bool)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        if (label[box[0]:box[2], box[1]:box[3]] > 0).any():
            keep[i] = False
    boxes = boxes[keep].reshape(-1, 4)
    idx_shuffle1 = pashuffle(boxes.shape[0], bg_shift_radio)
    idx_shuffle2 = np.random.permutation(idx_shuffle1)
    for i in range(idx_shuffle1.shape[0]):
        box1 = boxes[idx_shuffle1[i]]
        box2 = boxes[idx_shuffle2[i]]
        img_out[:, box1[0]:box1[2], box1[1]:box1[3]] = img[:, box2[0]:box2[2], box2[1]:box2[3]]
    return img_out


def elastic_transform(shape, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    # This function get indices only.
    if random_state is None:
        random_state = np.random.RandomState(None)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    return indices
