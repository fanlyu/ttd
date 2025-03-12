import yaml
import torch
import numpy as np
import hydra

from torch.utils.data import DataLoader
from core.stream.collate import DefaultCollateFunction
from core.stream.samplers import ExtendedSampler
from core.stream.dataset import load_dataset

import torchvision.transforms as T


class UPLStream():

    def __init__(self, dataset, config):

        self.dataset = dataset

        self.num_tasks = config.dataset.num_tasks
        self.num_classes = config.dataset.num_classes
        self.stream_size = config.dataset.stream_size

        self.task_classes = config.dataset.task_classes
        self.task_order = config.dataset.task_order
        self.stream_bs = config.dataset.stream_bs
        self.task_sizes = config.dataset.task_sizes
        self.t0_size = config.dataset.t0_size

        self.transform = T.Compose([T.Resize((config.dataset.img_size, config.dataset.img_size)), T.ToTensor(), T.Normalize(config.dataset.mean, config.dataset.std)])

        self.test_size = config.dataset.test_size
        self.task_ids = []

        for t, task in enumerate(self.task_order):
            if t == 0:
                continue
            self.task_ids += np.repeat(t, self.stream_size * len(task)).tolist()
        
        self.stream_inds = []
        self.super_loaders = []
        self.test_loaders = []

        self._index = 0
    
    def __next__(self):
        if self._index < len(self.stream_inds):
            all_data = []
            labels = []
            tasks = []
            for _ in range(self.stream_bs):
                ind = self.stream_inds[self._index]
                data, label = self.dataset.__getitem__(ind)
                data = self.transform(data)
                all_data.append(data)
                labels.append(int(label))
                tasks.append(int(self.task_ids[self._index]))

                self._index += 1
            all_data = torch.stack(all_data)
            labels = torch.tensor(labels)
            

            return all_data, labels, tasks[0]
        else:
            raise StopIteration
  
    def __iter__(self):
        return self
    
    def eval_loaders(self, t):
        pass

class IncrementalUPLStream(UPLStream):

    def __init__(self, dataset, config):
        super(IncrementalUPLStream, self).__init__(dataset, config)
        self.stream_size = config.dataset.stream_size
        self.super_size = config.dataset.super_size

        ### Collect the T0 data
        inds = []
        task_sup = []
        task_evl = []
        for cls in np.array(self.task_order[0])[:self.t0_size]:
            y_cls = self.dataset.selected_names.index(cls)
            if hasattr(self.dataset, 'object_ids'):
                cls_inds_train = np.argwhere((self.dataset.labels == y_cls) & (self.dataset.object_ids > 0)).flatten()
                cls_inds_test = np.argwhere((self.dataset.labels == y_cls) & (self.dataset.object_ids == -1)).flatten()
            else:
                cls_inds_train = np.argwhere(self.dataset.labels == y_cls).flatten()
                cls_inds_test = cls_inds_train

            np.random.shuffle(cls_inds_train)
            #inds += cls_inds_train[:self.stream_size].tolist()
            inds += cls_inds_train[:int(self.stream_size * config.dataset.t0_factor)].tolist()
            task_sup += cls_inds_train[:self.stream_size][np.random.choice(self.stream_size, self.super_size, replace=False)].tolist()
            task_evl += cls_inds_test[-self.test_size:].tolist()

        self.pretrain_dataloader = DataLoader(self.dataset, 
                        sampler=ExtendedSampler(inds, shuffle=True, repeats=config.model.init_epochs),
                        collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std),
                        batch_size=config.dataset.t0_bs, num_workers=12)
    
        self.super_loaders.append(
                DataLoader(self.dataset, 
                    sampler=ExtendedSampler(task_sup.copy(), shuffle=False),
                    collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                    batch_size=1)
                )                 

        self.test_loaders.append(
                DataLoader(self.dataset, 
                    sampler=ExtendedSampler(task_evl.copy(), shuffle=False),
                    collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                    batch_size=1)
                )

        y_cls = 0
        for t, task in enumerate(self.task_order):
            task_inds = []
            if t == 0:
                continue
            for cls in task:
                y_cls = self.dataset.selected_names.index(cls)
                if hasattr(self.dataset, 'object_ids'):
                    cls_inds_train = np.argwhere((self.dataset.labels == y_cls) & (self.dataset.object_ids > 0)).flatten()
                    cls_inds_test = np.argwhere((self.dataset.labels == y_cls) & (self.dataset.object_ids == -1)).flatten()
                else:
                    cls_inds_train = np.argwhere(self.dataset.labels == y_cls).flatten()
                    cls_inds_test = cls_inds_train

                np.random.shuffle(cls_inds_train)
                task_inds += cls_inds_train[:self.stream_size].tolist()
                task_sup += cls_inds_train[:self.stream_size][np.random.choice(self.stream_size, self.super_size, replace=False)].tolist()
                task_evl += cls_inds_test[-self.test_size:].tolist()

            np.random.shuffle(task_inds)
            self.stream_inds += task_inds
            self.super_loaders.append(
                    DataLoader(self.dataset, 
                        sampler=ExtendedSampler(task_sup.copy(), shuffle=False),
                        collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                        batch_size=1)
                    )                 

            self.test_loaders.append(
                    DataLoader(self.dataset, 
                        sampler=ExtendedSampler(task_evl.copy(), shuffle=False),
                        collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                        batch_size=1)
                    )

    def eval_loaders(self, t):
        return self.super_loaders[t], self.test_loaders[t]

    def task_length(self, t):
        return self.stream_size * len(self.task_order[t])
    
    def task_bounds(self, eval_freq):
        return [0, 1] + [eval_freq + (i * eval_freq) + 1 for i in range(self.num_tasks-2)]
    
    def eval_times(self, eval_freq):
        eval_times = []
        sep = self.task_length(1) // (eval_freq)
        for i in range(self.num_tasks * eval_freq):
            eval_times.append((sep // 2) + (sep * i))

        return eval_times, sep
    
    def __len__(self):
        return sum([self.task_length(i) for i in range(1, self.num_tasks)])


class BlurredUPLStream(UPLStream):

    def __init__(self, dataset, config):
        super(BlurredUPLStream, self).__init__(dataset, config)
        self.stream_size = config.dataset.stream_size
        self.super_size = config.dataset.super_size
        self.blur_size = config.scenario.blur_size

        ### Collect the T0 data
        inds = []
        task_sup = []
        task_evl = []
        for cls in np.array(self.task_order[0])[:self.t0_size]:
            y_cls = self.dataset.selected_names.index(cls)
            if hasattr(self.dataset, 'object_ids'):
                cls_inds_train = np.argwhere((self.dataset.labels == y_cls) & (self.dataset.object_ids > 0)).flatten()
                cls_inds_test = np.argwhere((self.dataset.labels == y_cls) & (self.dataset.object_ids == -1)).flatten()
            else:
                cls_inds_train = np.argwhere(self.dataset.labels == y_cls).flatten()
                cls_inds_test = cls_inds_train

            np.random.shuffle(cls_inds_train)
            #inds += cls_inds_train[:self.stream_size].tolist()
            inds += cls_inds_train[:int(self.stream_size * config.dataset.t0_factor)].tolist()
            task_sup += cls_inds_train[:self.stream_size][np.random.choice(self.stream_size, self.super_size, replace=False)].tolist()
            task_evl += cls_inds_test[-self.test_size:].tolist()

        self.pretrain_dataloader = DataLoader(self.dataset, 
                        sampler=ExtendedSampler(inds, shuffle=True, repeats=config.model.init_epochs),
                        collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std),
                        batch_size=config.dataset.t0_bs, num_workers=12)
    
        self.super_loaders.append(
                DataLoader(self.dataset, 
                    sampler=ExtendedSampler(task_sup.copy(), shuffle=False),
                    collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                    batch_size=1)
                )                 

        self.test_loaders.append(
                DataLoader(self.dataset, 
                    sampler=ExtendedSampler(task_evl.copy(), shuffle=False),
                    collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                    batch_size=1)
                )

        y_cls = 0
        for t, task in enumerate(self.task_order):
            task_inds = []
            if t == 0:
                continue
            for cls in task:
                y_cls = self.dataset.selected_names.index(cls)
                if hasattr(self.dataset, 'object_ids'):
                    cls_inds_train = np.argwhere((self.dataset.labels == y_cls) & (self.dataset.object_ids > 0)).flatten()
                    cls_inds_test = np.argwhere((self.dataset.labels == y_cls) & (self.dataset.object_ids == -1)).flatten()
                else:
                    cls_inds_train = np.argwhere(self.dataset.labels == y_cls).flatten()
                    cls_inds_test = cls_inds_train

                np.random.shuffle(cls_inds_train)
                task_inds += cls_inds_train[:self.stream_size].tolist()
                task_sup += cls_inds_train[:self.stream_size][np.random.choice(self.stream_size, self.super_size, replace=False)].tolist()
                task_evl += cls_inds_test[-self.test_size:].tolist()

            np.random.shuffle(task_inds)
            self.stream_inds += task_inds

            self.super_loaders.append(
                    DataLoader(self.dataset, 
                        sampler=ExtendedSampler(task_sup.copy(), shuffle=False),
                        collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                        batch_size=1)
                    )                 

            self.test_loaders.append(
                    DataLoader(self.dataset, 
                        sampler=ExtendedSampler(task_evl.copy(), shuffle=False),
                        collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                        batch_size=1)
                    )
            
        self.stream_inds = np.array(self.stream_inds)

        task_size = self.task_sizes[t] * self.stream_size
        blur_region = int(self.blur_size * task_size)
        big_bias = int(blur_region * 0.5)
        split = int(blur_region * 0.333)
        small_bias = int(blur_region * 0.166)
        for i in range(1, self.num_tasks-1):
            bound = i * task_size
            prior_task_inds = self.stream_inds[bound-blur_region:bound]
            post_task_inds = self.stream_inds[bound:bound+blur_region]

            blur_1 = np.array(prior_task_inds[:big_bias].tolist() + post_task_inds[:small_bias].tolist())
            np.random.shuffle(blur_1)
            blur_2 = np.array(prior_task_inds[big_bias:big_bias+split].tolist() + post_task_inds[small_bias:small_bias+split].tolist())
            np.random.shuffle(blur_2)
            blur_3 = np.array(prior_task_inds[big_bias+split:].tolist() + post_task_inds[small_bias+split:].tolist())
            np.random.shuffle(blur_3)

            self.stream_inds[bound-blur_region:bound+blur_region] = np.concatenate([blur_1, blur_2, blur_3])

    def eval_loaders(self, t):
        return self.super_loaders[t], self.test_loaders[t]

    def task_length(self, t):
        return self.stream_size * len(self.task_order[t])
    
    def task_bounds(self, eval_freq):
        return [0, 1] + [eval_freq + (i * eval_freq) + 1 for i in range(self.num_tasks-2)]
    
    def eval_times(self, eval_freq):
        eval_times = []
        sep = self.task_length(1) // (eval_freq)
        for i in range(self.num_tasks * eval_freq):
            eval_times.append((sep // 2) + (sep * i))

        return eval_times, sep
    
    def __len__(self):
        return sum([self.task_length(i) for i in range(1, self.num_tasks)])
    
class DynamicUPLStream(UPLStream):

    def __init__(self, dataset, config):
        super(DynamicUPLStream, self).__init__(dataset, config)

        self.stream_size = config.dataset.stream_size
        self.super_size = config.dataset.super_size

        ### Collect the T0 data
        inds = []
        task_sup = []
        task_evl = []
        for cls in np.array(self.task_order[0])[:self.t0_size]:
            y_cls = self.dataset.selected_names.index(cls)
            cls_inds = np.argwhere(self.dataset.labels == y_cls).flatten()
            
            inds += cls_inds[np.random.choice(len(cls_inds), self.stream_size, replace=False)].tolist()
            task_sup += cls_inds[np.random.choice(len(cls_inds), self.super_size)].tolist()
            eval_inds = np.argwhere((self.dataset.labels == y_cls) & (self.dataset.object_ids == -1)).flatten()
            task_evl += eval_inds[np.random.choice(len(eval_inds), self.test_size, replace=False)].tolist()
        

        self.pretrain_dataloader = DataLoader(self.dataset, 
                        sampler=ExtendedSampler(inds, shuffle=True, repeats=config.model.init_epochs),
                        collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std),
                        batch_size=config.dataset.t0_bs, num_workers=12)
    
        self.super_loaders.append(
                DataLoader(self.dataset, 
                    sampler=ExtendedSampler(task_sup.copy(), shuffle=False),
                    collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                    batch_size=1)
                )                 

        self.test_loaders.append(
                DataLoader(self.dataset, 
                    sampler=ExtendedSampler(task_evl.copy(), shuffle=False),
                    collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                    batch_size=1)
                )

        y_cls = 0
        for t, task in enumerate(self.task_order):
            task_inds = []
            if t == 0:
                continue
            for cls in task:
                y_cls = self.dataset.selected_names.index(cls)
                cls_inds = np.argwhere(self.dataset.labels == y_cls).flatten().tolist()
                obj_ids = np.unique(self.dataset.object_ids[cls_inds])
                frame_per_obj = (self.stream_size // (len(obj_ids) - 1)) + 1
                temp_stream = []
                temp_eval = []
                for obj in obj_ids:
                    obj_inds = np.argwhere((self.dataset.object_ids == obj) & (self.dataset.labels == y_cls)).flatten()
                    if obj == -1:
                        temp_eval += obj_inds.tolist()
                    else:
                        frame_pos = self.dataset.frame_pos[obj_inds[:frame_per_obj]]
                        sorted_inds = np.argsort(frame_pos)
                        temp_stream += obj_inds[sorted_inds].tolist()
                
                task_inds += np.array(temp_stream)[:self.stream_size].tolist()               
                task_evl += np.array(temp_eval)[np.random.choice(len(temp_eval), self.test_size, replace=False)].tolist()
                task_sup += np.array(temp_stream)[np.random.choice(len(temp_stream), self.super_size, replace=False)].tolist()
                

            #np.random.shuffle(task_inds)
            self.stream_inds += task_inds

            print(len(task_sup))
            print(len(task_evl))
            self.super_loaders.append(
                    DataLoader(self.dataset, 
                        sampler=ExtendedSampler(task_sup.copy(), shuffle=False),
                        collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                        batch_size=1)
                    )                 

            self.test_loaders.append(
                    DataLoader(self.dataset, 
                        sampler=ExtendedSampler(task_evl.copy(), shuffle=False),
                        collate_fn=DefaultCollateFunction(config.dataset.img_size, config.dataset.mean, config.dataset.std), 
                        batch_size=1)
                    )

    def eval_loaders(self, t):
        return self.super_loaders[t], self.test_loaders[t]
        
    def task_length(self, t):
        return self.stream_size * len(self.task_order[t])
    
    def task_bounds(self, eval_freq):
        return [0, 1] + [eval_freq + (i * eval_freq) + 1 for i in range(self.num_tasks-2)]
    
    def eval_times(self, eval_freq):
        eval_times = []
        sep = self.task_length(1) // (eval_freq)
        for i in range(self.num_tasks * eval_freq):
            eval_times.append((sep // 2) + (sep * i))

        return eval_times, sep
    
    def __len__(self):
        return sum([self.task_length(i) for i in range(1, self.num_tasks)])
    

def load_stream(config):

    dataset = load_dataset(config)

    if config.scenario.name == 'incremental':
        stream = IncrementalUPLStream(dataset, config)
    elif config.scenario.name == 'dynamic':
        stream = DynamicUPLStream(dataset, config)
    elif config.scenario.name == 'blurred':
        stream = BlurredUPLStream(dataset, config)

    return stream     
