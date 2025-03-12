import numpy as np
import copy
import os

from tqdm import tqdm
import torch, torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from core.models.scale.losses import SupConLoss, IRDLoss, similarity_mask_new, similarity_mask_old
# from core.stream.collate import SCALECollateFunction
from core.utils import smart_dir
from core.stream.dataset import NumpyDataset

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import contingency_matrix

from sklearn.decomposition import PCA

class PCAModel():

    def __init__(self, config):
        
        # declare properties
        self.name = 'SCALE'

        # extract scenario configs
        self.im_size = config.dataset.img_size
        self.num_c = config.dataset.channels
        self.seed = config.seed
        self.n_workers = config.model.n_workers
        self.num_tasks = config.dataset.num_tasks

        if config.model.pretrained:
            self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).to('cuda')
            self.backbone.fc = torch.nn.Identity()
        else:
            self.backbone = None


        # Cluster
        self.k_scale = config.model.k_scale
        self.components = config.model.components


        # Logging
        self.dataset = config.dataset.name
        self.log = config.log
        self.pretrain_log = config.pretrain_log



        # stam init
        self.init_layers()

    def init_layers(self):
        # random seed
        np.random.seed(self.seed)

    def pretrain(self, loader):
        pass


    def __call__(self, x, t):
        pass

        
    def eval(self, sup_loader, eval_loader, task, it=None):

        print('Supervising...')
        self.supervise(sup_loader, task)

        print('Classifying...')
        class_acc, class_acc_pc = self.classify(eval_loader, task)

        print('Clustering...')
        clust_acc, clust_acc_pc = self.cluster(eval_loader, task)

        return class_acc, class_acc_pc, clust_acc, clust_acc_pc
    
    def supervise(self, loader):
        
        embeddings = []
        labels = []
        for (x, y) in enumerate(loader):
            if self.backbone != None:
                embeddings.append(self.backbone(x.to('cuda')).detach().cpu().numpy())
            else:
                embeddings.append(x.reshape(x.shape[0], -1).numpy())
            labels.append(y.numpy())

        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)

        if self.backbone == None:
            embeddings = PCA(n_components=self.components).fit_transform(embeddings)


        self.knn_1 = KNeighborsClassifier(n_neighbors=1)
        self.knn_1.fit(embeddings, labels)

        self.knn_3 = KNeighborsClassifier(n_neighbors=3)
        self.knn_3.fit(embeddings, labels)

        self.knn_5 = KNeighborsClassifier(n_neighbors=5)
        self.knn_5.fit(embeddings, labels)

    # call classification function
    def classify(self, x, y):
        # embeddings = []
        # labels = []

        
        embeddings = x
        y_true = y
        
        # if self.backbone != None:
        #     embeddings.append(self.backbone(x.to('cuda')).detach().cpu().numpy())
        # else:

        # embeddings.append(x.reshape(x.shape[0], -1).cpu().numpy())
        # labels.append(y)

        # embeddings = np.concatenate(embeddings)
        # y_true = np.concatenate(labels)

        # if self.backbone == None:
        #     embeddings = PCA(n_components=self.components).fit_transform(embeddings)


        #y_pred_1 = self.knn_1.predict(embeddings)
        #y_pred_3 = self.knn_3.predict(embeddings)
        
        y_pred_5 = self.knn_5.predict(embeddings)

        num_classes = len(np.unique(y_true))
        acc = np.sum(np.array(y_true) == np.array(y_pred_5)) / len(y_pred_5)

        acc_pc = np.zeros(num_classes)
        for y in range(num_classes):
            inds_y = np.argwhere(y_true == y).flatten()
            acc_pc[y] = np.sum(np.array(y_true[inds_y]) == np.array(y_pred_5[inds_y])) / len(y_true[inds_y])

        print('Classification Acc: ', acc*100)
        
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
        
        #self.model_stats['class_acc'].append(acc)

        return acc, pc_acc
    

    # cluster
    def cluster(self, loader):
        
        embeddings = []
        labels = []
        for it, (x, y) in enumerate(loader):
            if self.backbone != None:
                embeddings.append(self.backbone(x.to('cuda')).detach().cpu().numpy())
            else:
                embeddings.append(x.reshape(x.shape[0], -1).numpy())
            labels.append(y.numpy())

        embeddings = np.concatenate(embeddings)
        y_true = np.concatenate(labels)

        if self.backbone == None:
            embeddings = PCA(n_components=self.components).fit_transform(embeddings)

        num_classes = len(np.unique(y_true)) * self.k_scale


        y_pred = KMeans(num_classes).fit_predict(embeddings)

        acc_total, acc_perclass = self.purity_score(y_true, y_pred)

        print('Clust Purity: ', acc_total * 100)


        return acc_total * 100, acc_perclass * 100


    def plots(self):
        return



