import os, time
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

import torchvision

class ImageNetDataset(Dataset):
    def __init__(self, data_dir, config, transform=None):
        self.transform = transform

        self.data = []
        self.labels = []
        class_names = np.sort([d for d in os.listdir(data_dir + 'train/')])
        if config.scenario.name == 'random':
            inds = np.random.choice(len(class_names), config.dataset.num_classes, replace=False).flatten()
            self.selected_classes = class_names[inds].tolist()
        else:
            self.selected_classes = []
            self.selected_names = []
            classes = {v: k for k, v in config.dataset.task_classes.items()}
            for task in config.dataset.task_order:
                for cls in task:
                    self.selected_classes.append(classes[cls])
                    self.selected_names.append(cls)


        for _, clas in enumerate(class_names):
            if clas in self.selected_classes:
                y = self.selected_classes.index(clas)
                for img in os.listdir(data_dir + 'train/' + clas):
                    self.data.append(data_dir + 'train/' + clas + '/' + img)
                    self.labels.append(y)

                for img in os.listdir(data_dir + 'val/' + clas):
                    self.data.append(data_dir + 'val/' + clas + '/' + img)
                    self.labels.append(y)

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        f_name = self.data[index]
        label = self.labels[index]

        img = Image.open(f_name).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
    

class CelebA_HQ(Dataset):
    def __init__(self, data_dir, config, transform=None):
        self.transform = transform

        self.data = []
        self.labels = []
        class_names = np.sort([d for d in os.listdir(data_dir + 'train/')])
        if config.scenario.name == 'random':
            inds = np.random.choice(len(class_names), config.dataset.num_classes, replace=False).flatten()
            self.selected_classes = class_names[inds].tolist()
        else:
            self.selected_classes = []
            self.selected_names = []
            classes = {v: k for k, v in config.dataset.task_classes.items()}
            for task in config.dataset.task_order:
                for cls in task:
                    self.selected_classes.append(classes[cls])
                    self.selected_names.append(cls)


        for _, clas in enumerate(class_names):
            if clas in self.selected_classes:
                y = self.selected_classes.index(clas)
                for img in os.listdir(data_dir + 'train/' + clas):
                    self.data.append(data_dir + 'train/' + clas + '/' + img)
                    self.labels.append(y)

                for img in os.listdir(data_dir + 'test/' + clas):
                    self.data.append(data_dir + 'test/' + clas + '/' + img)
                    self.labels.append(y)

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        f_name = self.data[index]
        label = self.labels[index]

        img = Image.open(f_name).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
    
class Food101Dataset(Dataset):
    def __init__(self, data_dir, config, transform=None):
        self.transform = transform

        self.data = []
        self.labels = []
        self.object_ids = []

        class_names = np.sort([d for d in os.listdir(data_dir + 'train/')])
        if config.scenario.name == 'random':
            inds = np.random.choice(len(class_names), config.dataset.num_classes, replace=False).flatten()
            self.selected_classes = class_names[inds].tolist()
        else:
            self.selected_classes = []
            self.selected_names = []
            classes = {v: k for k, v in config.dataset.task_classes.items()}
            for task in config.dataset.task_order:
                for cls in task:
                    self.selected_classes.append(classes[cls])
                    self.selected_names.append(cls)

        for _, clas in enumerate(class_names):
            if clas in self.selected_classes:
                y = self.selected_classes.index(clas)
                for img in os.listdir(data_dir + 'train/' + clas):
                    self.data.append(data_dir + 'train/' + clas + '/' + img)
                    self.labels.append(y)
                    self.object_ids.append(2)

                for img in os.listdir(data_dir + 'val/' + clas):
                    self.data.append(data_dir + 'val/' + clas + '/' + img)
                    self.labels.append(y)
                    self.object_ids.append(-1)

        self.labels = np.array(self.labels)
        self.object_ids = np.array(self.object_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        f_name = self.data[index]
        label = self.labels[index]

        img = Image.open(f_name).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

class AWA(Dataset):
    def __init__(self, data_dir, config, transform=None):
        self.transform = transform

        self.data = []
        self.labels = []
        class_names = np.sort([d for d in os.listdir(data_dir + 'JPEGImages/')])

        if config.scenario.name == 'random':
            inds = np.random.choice(config.dataset.num_classes, len(class_names), replace=False).flatten()
            self.selected_classes = class_names[inds].tolist()
        else:
            self.selected_classes = []
            self.selected_names = []
            classes = {v: k for k, v in config.dataset.task_classes.items()}
            for task in config.dataset.task_order:
                for cls in task:
                    self.selected_classes.append(classes[cls])
                    self.selected_names.append(cls)


        for _, clas in enumerate(class_names):
            if clas in self.selected_classes:
                y = self.selected_classes.index(clas)
                for img in os.listdir(data_dir + 'JPEGImages/' + clas):
                    self.data.append(data_dir + 'JPEGImages/' + clas + '/' + img)
                    self.labels.append(y)

        self.labels = np.array(self.labels)
        print(np.unique(self.labels, return_counts=True))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        f_name = self.data[index]
        label = self.labels[index]

        img = Image.open(f_name).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
    
class Stream51Dataset(Dataset):
    def __init__(self, data_dir, config, transform=None):
        self.transform = transform

        self.data = []
        self.labels = []
        self.object_ids = []
        self.frame_pos = []
        class_names = np.sort([d for d in os.listdir(data_dir + 'train/')])

        if config.scenario.name == 'random':
            inds = np.random.choice(config.dataset.num_classes, len(class_names), replace=False).flatten()
            self.selected_classes = class_names[inds].tolist()
        else:
            self.selected_classes = []
            self.selected_names = []
            classes = {v: k for k, v in config.dataset.task_classes.items()}
            for task in config.dataset.task_order:
                for cls in task:
                    self.selected_classes.append(classes[cls])
                    self.selected_names.append(cls)
        

        for _, clas in enumerate(class_names):
            if clas in self.selected_classes:
                y = self.selected_classes.index(clas)
                for img in os.listdir(data_dir + 'train/' + clas):
                    self.data.append(data_dir + 'train/' + clas + '/' + img)
                    self.object_ids.append(int(img.split('_')[0]))
                    self.frame_pos.append(int(img.split('_')[2].split('.')[0]))
                    self.labels.append(y)

        for _, clas in enumerate(class_names):
            if clas in self.selected_classes:
                print(clas)
                y = self.selected_classes.index(clas)
                evl_id = int(clas.split('-')[0]) - 1
                print(evl_id)
                for id in range(evl_id * 50, (evl_id+1)*50):
                    l = len(str(id))
                    img = '000000'[:-l] + str(id) + '.jpg'
                    self.data.append(data_dir + 'test/' + img)
                    self.object_ids.append(-1)
                    self.frame_pos.append(-1)
                    self.labels.append(y)


        self.labels = np.array(self.labels)
        self.object_ids = np.array(self.object_ids)
        self.frame_pos = np.array(self.frame_pos)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        f_name = self.data[index]
        label = self.labels[index]
        #start_time = time.time()
        img = Image.open(f_name).convert('RGB')
        #end_time = time.time()
        #print('Loading Image Time: ', end_time - start_time, flush=True)
        if self.transform:
            img = self.transform(img)

        return img, label

class RGBObjects(Dataset):
    def __init__(self, data_dir, config, transform=None):
        self.transform = transform

        self.data = []
        self.labels = []
        self.object_ids = []
        self.frame_pos = []
        class_names = np.sort([d for d in os.listdir(data_dir)])

        if config.scenario.name == 'random':
            inds = np.random.choice(config.dataset.num_classes, len(class_names), replace=False).flatten()
            self.selected_classes = class_names[inds].tolist()
        else:
            self.selected_classes = []
            self.selected_names = []
            classes = {v: k for k, v in config.dataset.task_classes.items()}
            for task in config.dataset.task_order:
                for cls in task:
                    self.selected_classes.append(classes[cls])
                    self.selected_names.append(cls)
        
        for _, clas in enumerate(class_names):
            if clas in self.selected_classes:
                y = self.selected_classes.index(clas)
                obj_id = -1
                for obj in os.listdir(data_dir + '/' + clas):
                    for img in os.listdir(data_dir + '/' + clas + '/' + obj):
                        if img.split('_')[-1] == 'crop.png':
                            self.data.append(data_dir + '/' + clas + '/' + obj + '/' + img)
                            self.object_ids.append(obj_id)
                            self.frame_pos.append(int(img.split('_')[3]))
                            self.labels.append(y)

                    obj_id += 1


        self.labels = np.array(self.labels)
        self.object_ids = np.array(self.object_ids)
        self.frame_pos = np.array(self.frame_pos)

        print(np.unique(self.labels, return_counts=True))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        f_name = self.data[index]
        label = self.labels[index]

        img = Image.open(f_name).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
    

class NumpyDataset(Dataset):

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def add_data(self, x, y):
        self.data = torch.cat((self.data, x))
        self.labels = np.concatenate((self.labels, y))

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return img, label


def load_dataset(config):

    if 'imagenet' in config.dataset.name:
        dataset = ImageNetDataset('datasets/imagenet/', 
                                    config)


    elif 'stream51' in config.dataset.name:
        dataset = Stream51Dataset('datasets/stream51/', 
                                    config)

    elif 'rgb-objects' in config.dataset.name:
        dataset = RGBObjects('datasets/rgb-objects',
                             config)
        
    elif 'food-101' in config.dataset.name:
        dataset = Food101Dataset('datasets/food-101/', config)

    elif 'awa' in config.dataset.name:
        dataset = AWA('datasets/animals_with_attributes/', config)

    elif 'celeba' in config.dataset.name:
        dataset = CelebA_HQ('datasets/CelebA_HQ/', config)
    
    elif 'places' in config.dataset.name:
        dataset = ImageNetDataset('datasets/places365_standard/', config)

    return dataset



