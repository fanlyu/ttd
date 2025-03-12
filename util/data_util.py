# -----------------------------------------------------------------------------
# Functions for data utility
# -----------------------------------------------------------------------------
import os
import json
from random import shuffle

import torch
import numpy as np
import pandas as pd
from glob import glob
from torchvision import transforms
from scipy import io as mat_io

from util.util import info


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class StrongWeakView(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, strong_transform, weak_transform):
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform

    def __call__(self, x):
        return [self.weak_transform(x), self.strong_transform(x)]


def build_transform(mode, args):
    '''
    Return transformed image
    '''
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
        
    if mode == 'default':
        crop_pct = args.crop_pct
        transform = transforms.Compose([
            transforms.Resize(int(args.input_size / crop_pct), interpolation=args.interpolation),
            transforms.RandomCrop(args.input_size),
            transforms.RandomHorizontalFlip(p=0.5 if args.dataset != 'mnist' else 0),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    elif mode == 'weak':
        crop_pct = args.crop_pct
        transform = transforms.Compose([
            transforms.Resize(int(args.input_size / crop_pct), interpolation=args.interpolation),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    
    elif mode == 'test':
        crop_pct = args.crop_pct
        transform = transforms.Compose([
            transforms.Resize(int(args.input_size / crop_pct), interpolation=args.interpolation),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    
    else:
        raise ValueError('Transform mode: {} not supported for GCD continual training.'.format(mode))

    return transform


def get_strong_transform(args):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = args.interpolation
    strong_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return strong_transform


def load_cifar100_images(split):
    info(f"Loading cifar100 images {split} dataset...")
    ########################## DATASET PATH ##########################
    dataset_path = './data/cifar-100-images'
    ########################## DATASET PATH ##########################
    class_dict_path = os.path.join('config/cifar100/class_dict.txt')
    with open(class_dict_path, 'r') as f:
        class_dict = f.read()
        class_dict = class_dict.replace("\'", "\"")
    
    class_dict = json.loads(class_dict)

    # Load train and val images path + labels
    train_label, val_label = [], []

    train_list = glob(os.path.join(dataset_path, 'train/*/*.png'))
    val_list = glob(os.path.join(dataset_path, 'test/*/*.png'))

    for image_path in train_list:
        label = int(class_dict[os.path.split(image_path)[0].split('/')[-1]])
        train_label.append(label)
    
    for image_path in val_list:
        label = int(class_dict[os.path.split(image_path)[0].split('/')[-1]])
        val_label.append(label)

    return ((train_list, train_label), (val_list, val_label))

def load_tiny_imagenet_200(split):
    info(f"Loading Tiny ImageNet 200 {split} dataset ...")
    ########################## DATASET PATH ##########################
    dataset_path = './data/tiny-imagenet-200'
    ########################## DATASET PATH ##########################
    class_dict_path = os.path.join('config/tinyimagenet/class_dict.txt')
    with open(class_dict_path, 'r') as f:
        class_dict = f.read()
        class_dict = class_dict.replace("\'", "\"")
    
    class_dict = json.loads(class_dict)

    # Load train and val images path + labels
    train_label, val_label = [], []

    train_list = glob(os.path.join(dataset_path, 'train/*/*/*.JPEG'))
    val_list = glob(os.path.join(dataset_path, 'val/*/*.JPEG'))

    for image_path in train_list:
        label = int(class_dict[os.path.split(image_path)[0].split('/')[-2]])
        train_label.append(label)
    
    for image_path in val_list:
        label = int(class_dict[os.path.split(image_path)[0].split('/')[-1]])
        val_label.append(label)


    return ((train_list, train_label), (val_list, val_label))

def load_CUB_200(split):
    info(f"Loading CUB 200 {split} dataset ...")
    ########################## DATASET PATH ##########################
    dataset_path = './data/CUB/CUB_200_2011'
    ########################## DATASET PATH ##########################

    images = pd.read_csv(os.path.join(dataset_path, 'images.txt'), sep=' ', names=['img_id', 'filepath'])
    image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'])
    train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])

    data = images.merge(image_class_labels, on='img_id').merge(train_test_split, on='img_id')
    data['filepath'] = data['filepath'].apply(lambda x: os.path.join(dataset_path, 'images', x))

    # Split train and val
    train = data[data['is_training_img'] == 1]
    val = data[data['is_training_img'] == 0]

    # Load train and val images path + labels
    train_list = train['filepath'].values
    train_label = train['target'].values - 1 #CUB labels start from 1

    val_list = val['filepath'].values
    val_label = val['target'].values - 1 #CUB labels start from 1

    return ((train_list, train_label), (val_list, val_label))