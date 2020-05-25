"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from collections import defaultdict
import math
import logging
import datasets.ade20k_labels as ade20k_labels
import json
from config import cfg
import torchvision.transforms as transforms
import datasets.edge_utils as edge_utils
import cv2
import os
from pathlib import Path
import pickle

import time
from tqdm import tqdm
import bisect
import csv
import faulthandler
trainid_to_name = ade20k_labels.trainId2name
id_to_trainid = ade20k_labels.label2trainid
num_classes = 150
ignore_label = 255
root = cfg.DATASET.CITYSCAPES_DIR
palette = []
with open("output.csv") as merged:
    reader = csv.reader(merged, delimiter=",")
    for idx, row in enumerate(reader):
        palette.append(row[0])
        palette.append(row[1])
        palette.append(row[2])


zero_pad = 256 * 3 - len(palette)

for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def add_items(items, aug_items, cities, img_path, mask_path, mask_postfix, mode, maxSkip):
    items = find_match(cities, mode)
    return items
    

def find_match(cities, mode): 
    segm_imgs = []
    c=""
    if mode == "train":
        c = "training"
    elif mode == "val":
        c = "validation"
    segm_path = "/pless_nfs/home/mdt_/GSCNN/ADE20K_2016_07_26/images/" + c
    segm_imgs = [f.as_posix() for f in Path(segm_path).glob("**/*.png") if "seg" in f.as_posix() and c in f.as_posix()]
    
    
    from  builtins import any as b_any
    result = []
    result_seg = []
    import os
    import subprocess
    if os.path.exists("./tp.pickle"):
        with open("tp.pickle", "rb") as fp:
            result = pickle.load(fp)
    else:
        for img in range(len(cities)):  
            matching = []
            fn = (os.path.splitext(os.path.basename(cities[img]))[0]).split("_", 2)[2]
            q = 0
            for msk in segm_imgs:
                if fn in msk:
                    q = q+1
                    tp = (cities[img], msk)
                    result.append(tp)
    with open("tp.pickle", "wb+") as fp:   #Pickling   
        pickle.dump(result, fp)

    return result

def sorter(val):
    return os.path.splitext(os.path.basename(val))[0]

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def binary_search(Array, Search_Term):
    n = len(Array)
    L = 0
    R = n-1

    Search_Term = int(sorter(Search_Term))
    
    while L <= R:
        mid = ((L+R)//2)
        if int(sorter(Array[mid])) < Search_Term:
            L = mid + 1
        elif int(sorter(Array[mid])) > Search_Term:
            R = mid - 1
        else:
            return mid
    return -1

def make_cv_splits(img_dir_name):
    '''
    Create splits of train/val data.
    A split is a lists of cities.
    split0 is aligned with the default Cityscapes train/val.
    '''
    trn_path = os.path.join(root, img_dir_name, 'training')
    val_path = os.path.join(root, img_dir_name, 'validation')
    t = []
    v =[]
    if os.path.exists("./tr.pickle") and os.path.exists("./val.pickle"):
        with open("tr.pickle", "rb") as fp:
            print("retirieved pickle for train")
            t = pickle.load(fp)
        with open("val.pickle", "rb") as fp:
            v = pickle.load(fp)
    else:
        train_path = "/pless_nfs/home/mdt_/GSCNN/ADE20K_2016_07_26/images/training"
        v_path = "/pless_nfs/home/mdt_/GSCNN/ADE20K_2016_07_26/images/validation"
        t = [f.as_posix() for f in Path(train_path).glob("**/*.jpg")]
        v = [f.as_posix() for f in Path(v_path).glob("**/*.jpg")]
        with open("tr.pickle", "wb+") as fp:   #Pickling   
            pickle.dump(t, fp)
        with open("val.pickle", "wb+") as fp:   #Pickling   
            pickle.dump(v, fp)
    trn_cities = sorted(t)

    all_cities = t + v
    num_val_cities = len(v)
    num_cities = len(all_cities)
    num_t_cities = len(t)

    t = np.array_split(np.array(t), cfg.DATASET.CV_SPLITS)
    v = np.array_split(np.array(v), cfg.DATASET.CV_SPLITS)

    cv_splits = []
    for split_idx in range(cfg.DATASET.CV_SPLITS):
        split = {}
        split['train'] = []
        split['val'] = []
        split['val'].extend(v[split_idx].tolist())
        split['train'].extend(t[split_idx].tolist())
        cv_splits.append(split)

    
    
    return cv_splits


def make_split_coarse(img_path):
    '''
    Create a train/val split for coarse
    return: city split in train
    '''
    print(img_path)
    all_cities = os.listdir(img_path)
    all_cities = sorted(all_cities)  # needs to always be the same
    val_cities = [] # Can manually set cities to not be included into train split

    split = {}
    split['val'] = val_cities
    split['train'] = [c for c in all_cities if c not in val_cities]
    return split

def make_test_split(img_dir_name):
    test_path = os.path.join(root, img_dir_name, 'leftImg8bit', 'test')
    test_cities = ['test/' + c for c in os.listdir(test_path)]

    return test_cities


def make_dataset(quality, mode, maxSkip=0, fine_coarse_mult=6, cv_split=0):
    '''
    Assemble list of images + mask files

    fine -   modes: train/val/test/trainval    cv:0,1,2
    coarse - modes: train/val                  cv:na

    path examples:
    leftImg8bit_trainextra/leftImg8bit/train_extra/augsburg
    gtCoarse/gtCoarse/train_extra/augsburg
    '''
    items = []
    aug_items = []

    if quality == 'fine':
        assert mode in ['train', 'val', 'test', 'trainval']
        img_dir_name = 'Hotels-50k'
        img_path = os.path.join(root, img_dir_name)
        mask_path = os.path.join(root, 'annotations', 'h', 'hotel')
        mask_postfix = '_gtFine_labelIds.png'
        cv_splits = make_cv_splits(img_dir_name)
        if mode == 'trainval':
            modes = ['train', 'val']
        else:
            modes = [mode]
        for mode in modes:
            if mode == 'test':
                cv_splits = make_test_split(img_dir_name)
                items = add_items(items, cv_splits, img_path, mask_path,
                      mask_postfix)
            else:
                print(cv_split)
                logging.info('{} fine cities: '.format(mode) + str(len(cv_splits[cv_split][mode])))

                items = add_items(items, aug_items, cv_splits[cv_split][mode], img_path, mask_path,
                      mask_postfix, mode, maxSkip)
    else:
        raise 'unknown sun quality {}'.format(quality)
    logging.info('Sun-{}: {} images'.format(mode, len(items)+len(aug_items)))
    return items, aug_items


class ade20k(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform=None, sliding_crop=None,
                 transform=None, target_transform=None, dump_images=False,
                 cv_split=None, eval_mode=False, 
                 eval_scales=None, eval_flip=False):
        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.eval_mode = eval_mode
        self.eval_flip = eval_flip
        self.eval_scales = None
        if eval_scales != None:
            self.eval_scales = [float(scale) for scale in eval_scales.split(",")]

        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS, \
                'expected cv_split {} to be < CV_SPLITS {}'.format(
                    cv_split, cfg.DATASET.CV_SPLITS)
        else:
            self.cv_split = 0
        self.imgs, _ = make_dataset(quality, mode, self.maxSkip, cv_split=self.cv_split)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _eval_get_item(self, img, mask, scales, flip_bool):
        return_imgs = []
        for flip in range(int(flip_bool)+1):
            imgs = []
            if flip :
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for scale in scales:
                w,h = img.size
                target_w, target_h = int(w * scale), int(h * scale) 
                resize_img =img.resize((target_w, target_h))
                tensor_img = transforms.ToTensor()(resize_img)
                final_tensor = transforms.Normalize(*self.mean_std)(tensor_img)
                imgs.append(tensor_img)
            return_imgs.append(imgs)
        return return_imgs, mask
        


    def __getitem__(self, index):
        faulthandler.enable()

        img_path, mask_path = self.imgs[index]

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path).convert('LA')
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in id_to_trainid.items():
            mask_copy[mask == k] = v

        if self.eval_mode:
            return self._eval_get_item(img, mask_copy, self.eval_scales, self.eval_flip), img_name

        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image Transformations
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        _edgemap = mask.numpy()
        _edgemap = _edgemap[:, :, 0]
        _edgemap = edge_utils.mask_to_onehot(_edgemap, num_classes)
        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)

        edgemap = torch.from_numpy(_edgemap).float()
        
	# Debug
        if self.dump_images:
            outdir = '../../dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, img_name + '.png')
            out_msk_fn = os.path.join(outdir, img_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask))
            img.save(out_img_fn)
            mask_img.save(out_msk_fn)

        return img, mask, edgemap, img_name

    def __len__(self):
        return len(self.imgs)


def make_dataset_video():
    img_dir_name = 'leftImg8bit_demoVideo'
    img_path = os.path.join(root, img_dir_name, 'leftImg8bit/demoVideo')
    items = []
    categories = os.listdir(img_path)
    for c in categories[1:]:
        c_items = [name for name in
                   os.listdir(os.path.join(img_path, c))]
        for it in c_items:
            item = os.path.join(img_path, c)
            items.append(item)
    return items


class SunVideo(data.Dataset):

    def __init__(self, transform=None):
        self.imgs = make_dataset_video()
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.imgs)

