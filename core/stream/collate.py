import torch, torchvision
import torchvision.transforms as T
# from lightly.data import LightlyDataset

import torch.nn as nn

from typing import List, Tuple, Union, Optional

from PIL import Image

import numpy as np
import time

from numpy.lib import stride_tricks
import torchvision.utils as vutils

import time

class DefaultCollateFunction(nn.Module):

    def __init__(self, img_size, mean, std):

        super(DefaultCollateFunction, self).__init__()
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean, std)])

    def forward(self, batch: List[Tuple[Image.Image, int, str]]):
        imgs = []
        for item in batch:
            imgs.append(self.transform(item[0]))

        labels = torch.LongTensor([item[1] for item in batch])
        
        imgs = torch.stack(imgs)

        return imgs, labels

class ExtractPatches(nn.Module):

    def __init__(self, img_size, patch_size, transform1, transform2, stride=None):

        super(ExtractPatches, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        if stride == None:
            self.stride = patch_size
        else:
            self.stride = stride
        self.ch = 3

        self.resize = T.Resize((self.img_size, self.img_size))

        self.transform1 = transform1
        self.transform2 = transform2

        self.num_patches = int(np.power(np.floor((self.img_size - self.patch_size) / self.stride) + 1, 2))
        self.num_patch_axis = int(np.ceil((self.img_size - self.patch_size) / 1) + 1)

        
    def forward(self, im):
        im = self.resize(im)
        im = np.array(im)

        shape = (self.num_patch_axis, self.num_patch_axis, self.patch_size, self.patch_size, self.ch)
        strides = (im.strides[0], im.strides[1], im.strides[0], im.strides[1], im.strides[2])
        patches = stride_tricks.as_strided(im, shape=shape, strides=strides)
        patches = patches[range(0,self.num_patch_axis,self.stride),:][:,range(0,self.num_patch_axis,self.stride),:,:,:]
        patches = patches.reshape((self.num_patches, self.patch_size, self.patch_size, self.ch))


        batch1 = torch.stack([self.transform1(Image.fromarray(patches[i])) for i in range(self.num_patches)])
        if self.transform2 != None:
            batch2 = torch.stack([self.transform2(Image.fromarray(patches[i])) for i in range(self.num_patches)])
        else:
            batch2 = None

        return batch1, batch2

class PatchCollateFunction(nn.Module):
    """Base class for other collate implementations.
    Takes a batch of images as input and transforms each image into two
    different augmentations with the help of random transforms. The images are
    then concatenated such that the output batch is exactly twice the length
    of the input batch.
    Attributes:
        transform:
            A set of torchvision transforms which are randomly applied to
            each image.
    """

    def __init__(self, img_size, patch_size, transform1, transform2, stride=None):

        super(PatchCollateFunction, self).__init__()
        self.transform = ExtractPatches(img_size, patch_size, transform1, transform2, stride)
        self.basic_transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.augs = True

    def forward(self, batch: List[Tuple[Image.Image, int, str]]):
        """Turns a batch of tuples into a tuple of batches.
        Args:
            batch:
                A batch of tuples of images, labels, and filenames which
                is automatically provided if the dataloader is built from
                a LightlyDataset.
        Returns:
            A tuple of images, labels, and filenames. The images consist of
            two batches corresponding to the two transformations of the
            input images.
        Examples:
            >>> # define a random transformation and the collate function
            >>> transform = ... # some random augmentations
            >>> collate_fn = BaseCollateFunction(transform)
            >>>
            >>> # input is a batch of tuples (here, batch_size = 1)
            >>> input = [(img, 0, 'my-image.png')]
            >>> output = collate_fn(input)
            >>>
            >>> # output consists of two random transforms of the images,
            >>> # the labels, and the filenames in the batch
            >>> (img_t0, img_t1), label, filename = output
        """
        #start_time = time.time()
        batch_size = len(batch)

        # list of transformed images
        
        imgs_a = []
        imgs_b = []
        imgs_basic = []
        for item in batch:
            #s_t = time.time()
            if self.augs:
                bat_a, bat_b = self.transform(item[0])
                imgs_a.append(bat_a)
                imgs_b.append(bat_b)
            imgs_basic.append(self.basic_transform(item[0]))
            #e_t = time.time()
            #print('Transform Time: ', e_t - s_t, flush=True)

        labels = torch.LongTensor([item[1] for item in batch]).repeat_interleave(self.transform.num_patches)
        
        if self.augs:
            transforms = (torch.cat(imgs_a), torch.cat(imgs_b))
        else:
            transforms = None

        basics = torch.stack(imgs_basic)

        #end_time = time.time()
        #print('Time Taken: ', end_time - start_time, flush=True)
        return transforms, basics, labels
    

class SleepCollateFunction(nn.Module):
    """Base class for other collate implementations.
    Takes a batch of images as input and transforms each image into two
    different augmentations with the help of random transforms. The images are
    then concatenated such that the output batch is exactly twice the length
    of the input batch.
    Attributes:
        transform:
            A set of torchvision transforms which are randomly applied to
            each image.
    """

    def __init__(self, patch_size, transform1, transform2):

        super(SleepCollateFunction, self).__init__()
        self.transform1 = transform1
        self.transform2 = transform2

    def forward(self, batch):
        # list of transformed images

        imgs_a = []
        imgs_b = []
        for item in batch:
            imgs_a.append(self.transform1(item[0]))
            imgs_b.append(self.transform2(item[0]))


        labels = torch.LongTensor([item[1] for item in batch])
        
        # list of filenames
        fnames = [item[2] for item in batch]


        transforms = (torch.stack(imgs_a), torch.stack(imgs_b))



        return transforms, labels, fnames
    

class SCALECollateFunction(nn.Module):
    """Base class for other collate implementations.
    Takes a batch of images as input and transforms each image into two
    different augmentations with the help of random transforms. The images are
    then concatenated such that the output batch is exactly twice the length
    of the input batch.
    Attributes:
        transform:
            A set of torchvision transforms which are randomly applied to
            each image.
    """

    def __init__(self, transform):

        super(SCALECollateFunction, self).__init__()
        self.transform = transform

    def forward(self, batch):
        # list of transformed images

        imgs_a = []
        imgs_b = []
        for item in batch:
            imgs_a.append(self.transform(item[0]))
            imgs_b.append(self.transform(item[0]))


        labels = torch.LongTensor([item[1] for item in batch])
        


        transforms = (torch.stack(imgs_a), torch.stack(imgs_b))

        return transforms, labels
    
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        return (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


if __name__ == '__main__':


    pass

