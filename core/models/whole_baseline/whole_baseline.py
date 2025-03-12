import numpy as np
import copy
import os

from tqdm import tqdm
import torch, torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from core.models.scale.losses import SupConLoss, IRDLoss, similarity_mask_new, similarity_mask_old
from core.stream.collate import SCALECollateFunction
from core.utils import smart_dir
from core.stream.dataset import NumpyDataset
from core.stream.samplers import ExtendedSampler

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import contingency_matrix

class WholeBaseline():

    def __init__(self, config):
        
        # declare properties
        self.name = 'SCALE'

        # extract scenario configs
        self.im_size = config.dataset.img_size
        self.num_c = config.dataset.channels
        self.seed = config.seed
        self.n_workers = config.model.n_workers
        self.num_tasks = config.dataset.num_tasks

        # memory
        self.stm = []
        self.ltm = []

        self.stm_size = config.model.stm_size
        self.ltm_size = config.model.ltm_size

        # Cluster
        self.k_scale = config.model.k_scale


        # Encoder
        resnet = torchvision.models.resnet18().to('cuda')
        self.backbone = resnet
        self.backbone.fc = torch.nn.Identity()

        # Training
        self.init_epochs = config.model.epochs
        self.bs = config.model.bs
        #self.simil = config.model.simil
        #self.losses_distil = []
        #self.losses_contrast = []
        #self.losses = []
        #self.distill_power_1 = 0
        #self.distill_power = config.model.distil_power
        self.stream_train = config.model.stream_train
        self.sleep_freq = config.model.sleep_freq
        self.load_pretrain = config.model.load_pretrain

        # Augmentations
        self.base_transform = [
            transforms.RandomResizedCrop(size=self.im_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.6, 0.6, 0.6, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.dataset.mean, std=config.dataset.std),
        ]

        self.no_aug_transform = transforms.Compose([
            transforms.Resize((self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.dataset.mean, std=config.dataset.std)
        ])

        self.pretrain_transform = transforms.Compose(self.base_transform)
        self.stream_transform = transforms.Compose([transforms.Normalize(mean=[-config.dataset.mean[i] / config.dataset.std[i] for i in range(3)] , std=[1 / config.dataset.std[i] for i in range(3)]), transforms.ToPILImage()] + self.base_transform)
                                                   

        self.pretrain_collate = SCALECollateFunction(self.pretrain_transform)
        self.mem_init_collate = SCALECollateFunction(self.no_aug_transform)
        self.sleep_collate = SCALECollateFunction(self.stream_transform)


        # Optim
        self.criterion = SupConLoss(config.model.bs, 'resnet18', config.model.temp, config.model.base_temp).to('cuda')
        self.optimizer = torch.optim.SGD(list(self.backbone.parameters()) + list(self.criterion.projector.parameters()), config.model.lr, config.model.momentum, config.model.wd)


        for param_group in self.optimizer.param_groups:
            param_group['lr'] = config.model.lr


        # Logging
        self.dataset = config.dataset.name
        self.log = config.log
        self.pretrain_log = config.pretrain_log
        self.step = 0
        self.sleeps = 0


        # stam init
        self.init_layers()

    def init_layers(self):
        
        # random seed
        np.random.seed(self.seed)

    def pretrain(self, loader):
        self.backbone.train()
        dataset = loader.dataset
        sampler = loader.sampler
        epochs = self.init_epochs

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.bs,
            collate_fn=self.pretrain_collate,
            drop_last=True,
            sampler=sampler,
            num_workers=self.n_workers,
        )

        log = f'logs/{self.pretrain_log}/{self.dataset}/task_0/saved_models/final_backbone.pkl'
        log2 = f'logs/{self.pretrain_log}/{self.dataset}/task_0/saved_models/final_projector.pkl'
        if self.load_pretrain and os.path.exists(log):
            print('Loading Pretrain')
            self.backbone.load_state_dict(torch.load(log))
            self.criterion.projector.load_state_dict(torch.load(log2))
        else:

            ite = 0
            loss_history = []
            total_loss = 0
            for batch, t in tqdm(trainloader):

                if ite % len(trainloader) == 0:
                    vutils.save_image(batch[0][:], smart_dir(f'logs/{self.log}/{self.dataset}/task_0') + 'aug_0.png', normalize=True)
                    vutils.save_image(batch[1][:], smart_dir(f'logs/{self.log}/{self.dataset}/task_0') + 'aug_1.png', normalize=True)

                self.optimizer.zero_grad()

                loss = self.criterion(self.backbone, self.backbone, batch[0].to('cuda'), batch[1].to('cuda'), None, None)
                
                loss.backward()

                total_loss += loss.item()

                self.optimizer.step()
                ite += 1

                if ite % (len(trainloader) // self.init_epochs) == 0:
                    avg_loss = total_loss / (len(trainloader) // self.init_epochs)
                    total_loss = 0
                    print(f'Epoch: [{ite}/{len(trainloader)}],   Loss: {avg_loss}')
                    loss_history.append(avg_loss)
                    plt.plot(loss_history)
                    plt.title('Average Loss over Training')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.savefig(smart_dir(f'logs/{self.log}/{self.dataset}/task_0/layer_{self.name}/figures') + 'loss_history.png')
                    plt.close()
                    
            torch.save(self.backbone.state_dict(), smart_dir(f'logs/{self.log}/{self.dataset}/task_0/saved_models') + f'final_backbone.pkl')
            torch.save(self.criterion.projector.state_dict(), smart_dir(f'logs/{self.log}/{self.dataset}/task_0/saved_models') + f'final_projector.pkl')

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.bs,
            collate_fn=self.mem_init_collate,
            drop_last=True,
            sampler=sampler,
            num_workers=self.n_workers,
        )

        it = 0
        for batch, t in trainloader:
            if it >= (len(trainloader) // self.init_epochs):
                break
            self.ltm.append(batch[0])
            it += 1

        self.ltm = torch.cat(self.ltm)
        inds = np.random.choice(len(self.ltm), self.ltm_size, replace=True).flatten()
        self.ltm = self.ltm[inds]

        self.backbone.eval()
        print('Done Initializing')

    def sleep(self):        
        self.backbone.train()
        print('Going to Sleep')

        self.stm = torch.cat(self.stm)
        inds = np.random.choice(len(self.stm), self.stm_size, replace=False).flatten()
        self.stm = self.stm[inds]
        data = torch.cat((self.stm, self.ltm))
        dataset = NumpyDataset(data, np.repeat(1, len(data)))
        self.ltm = torch.cat((self.ltm, self.stm[:self.ltm_size]))

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.bs,
            collate_fn=self.sleep_collate,
            drop_last=True,
            sampler=ExtendedSampler(np.arange(len(dataset.labels)), shuffle=True, repeats=self.init_epochs),
            num_workers=self.n_workers,
        )

        ite = 0
        loss_history = []
        #for epoch in range(self.init_epochs):
        total_loss = 0
        for batch, t in tqdm(trainloader):

            if ite % len(trainloader) == 0:
                vutils.save_image(batch[0][:], smart_dir(f'logs/{self.log}/{self.dataset}/sleep_{self.sleeps}') + 'aug_0.png', normalize=True)
                vutils.save_image(batch[1][:], smart_dir(f'logs/{self.log}/{self.dataset}/sleep_{self.sleeps}') + 'aug_1.png', normalize=True)

            self.optimizer.zero_grad()

            loss = self.criterion(self.backbone, self.backbone, batch[0].to('cuda'), batch[1].to('cuda'), None, None)
            
            loss.backward()

            total_loss += loss.item()

            self.optimizer.step()
            ite += 1


        if ite % (len(trainloader) // self.init_epochs) == 0:
            avg_loss = total_loss / (len(trainloader) // self.init_epochs)

            print(f'Epoch: [{ite}/{len(trainloader)}],   Loss: {avg_loss}')
            loss_history.append(avg_loss)
            plt.plot(loss_history)
            plt.title('Average Loss over Training')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.savefig(smart_dir(f'logs/{self.log}/{self.dataset}/sleep_{self.sleeps}/figures') + 'loss_history.png')
            plt.close()
            
        self.backbone.eval()
        self.stm = []
        self.sleeps += 1


    def __call__(self, x, t):

        self.stm.append(x)

        self.step += 1

        if (self.step - 1 + (self.sleep_freq // 2)) % self.sleep_freq == 0:
            self.sleep()

        
    def eval(self, sup_loader, eval_loader, task, it=None):

        print('Supervising...')
        self.supervise(sup_loader, task)

        print('Classifying...')
        class_acc, class_acc_pc = self.classify(eval_loader, task)

        print('Clustering...')
        clust_acc, clust_acc_pc = self.cluster(eval_loader, task)

        return class_acc, class_acc_pc, clust_acc, clust_acc_pc
    
    def supervise(self, loader, task):
        
        embeddings = []
        labels = []
        for it, (x, y) in enumerate(loader):
            embeddings.append(self.backbone(x.to('cuda')).detach().cpu().numpy())
            labels.append(y.numpy())

        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)

        self.knn_1 = KNeighborsClassifier(n_neighbors=1)
        self.knn_1.fit(embeddings, labels)

        self.knn_3 = KNeighborsClassifier(n_neighbors=3)
        self.knn_3.fit(embeddings, labels)

        self.knn_5 = KNeighborsClassifier(n_neighbors=5)
        self.knn_5.fit(embeddings, labels)

    # call classification function
    def classify(self, loader, task):
        embeddings = []
        labels = []
        for it, (x, y) in enumerate(loader):
            embeddings.append(self.backbone(x.to('cuda')).detach().cpu().numpy())
            labels.append(y.numpy())
        
        embeddings = np.concatenate(embeddings)
        y_true = np.concatenate(labels)   


        y_pred_1 = self.knn_1.predict(embeddings)
        y_pred_3 = self.knn_3.predict(embeddings)
        y_pred_5 = self.knn_5.predict(embeddings)

        num_classes = len(np.unique(y_true))
        acc1 = np.sum(np.array(y_true) == np.array(y_pred_1)) / len(y_pred_1)

        num_classes = len(np.unique(y_true))
        acc3 = np.sum(np.array(y_true) == np.array(y_pred_3)) / len(y_pred_3)

        num_classes = len(np.unique(y_true))
        acc5 = np.sum(np.array(y_true) == np.array(y_pred_5)) / len(y_pred_5)

        accs = [acc1, acc3, acc5]
        ind = np.argmax(accs)

        acc = accs[ind]

        if ind == 0:
            y_pred = y_pred_1
        elif ind == 1:
            y_pred = y_pred_3
        else:
            y_pred = y_pred_5

        acc_pc = np.zeros(num_classes)
        for y in range(num_classes):
            inds_y = np.argwhere(y_true == y).flatten()
            acc_pc[y] = np.sum(np.array(y_true[inds_y]) == np.array(y_pred[inds_y])) / len(y_true[inds_y])
        
        return acc * 100, acc_pc * 100
    
    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        cm = contingency_matrix(y_true, y_pred)

        num_classes = len(np.unique(y_true))

        cluster_labs = np.argmax(cm, axis=0)

        class_perf = np.zeros(num_classes)
        class_counts = np.zeros(num_classes)

        for i, lab in enumerate(cluster_labs):
            class_perf[lab] += cm[lab, i] / np.sum(cm[:, i])
            class_counts[lab] += 1

        for i in range(len(class_counts)):
            class_perf[i] /= class_counts[i]

        # return purity
        acc, pc_acc = np.sum(np.amax(cm, axis=0)) / np.sum(cm), class_perf

        return acc, pc_acc
    

    # cluster
    def cluster(self, loader, task):
        
        embeddings = []
        labels = []
        for it, (x, y) in enumerate(loader):
            embeddings.append(self.backbone(x.to('cuda')).detach().cpu().numpy())
            labels.append(y.numpy())
        
        embeddings = np.concatenate(embeddings)
        y_true = np.concatenate(labels) 
        num_classes = len(np.unique(y_true)) * self.k_scale


        y_pred = KMeans(num_classes).fit_predict(embeddings)

        acc_total, acc_perclass = self.purity_score(y_true, y_pred)

        return acc_total * 100, acc_perclass * 100


    def plots(self):
        return



