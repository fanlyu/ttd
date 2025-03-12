import tqdm, os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from sklearn.manifold import TSNE

from core.models.pcmc.pcmc_layer import Layer
from core.utils import *

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances

import torchvision.transforms as T
from sklearn.metrics.cluster import contingency_matrix

from scipy import stats


class PCMC():

    def __init__(self, config):

        # declare properties
        self.name = 'PCMC'

        # extract scenario configs
        self.im_size = config.dataset.img_size
        self.num_c = config.dataset.channels
        self.seed = config.seed

        # extract pcmc configs
        self.num_layers = config.model.num_layers

        self.layers = []
        for l in range(self.num_layers):
            self.layers.append(Layer(l, config))
        # pcmc init
        self.initialized = False
        self.pretrain_only = config.model.pretrain_only
        self.task = 0
        self.step = 1

        # Logging
        self.dataset = config.dataset.name
        self.log = config.log


    def init_layers(self):
        # random seed
        np.random.seed(self.seed)

    def pretrain(self, dataloader):
        for layer in self.layers:
            layer.pretrain(dataloader)
        self.task += 1

    def __call__(self, data, t):
        for layer in self.layers:
            layer(data, t)  

    # get percent class informative centroids
    def get_ci(self):
        scores = []
        scores_pc = []
        for layer in self.layers:
            score, score_pc = layer.get_ci()
            scores.append(score)
            scores_pc.append(score_pc)

        return np.array(scores), np.stack(score_pc)

    # given labeled x, associate class information with pcmc centroids
    def supervise(self, loader, task, iter):
        X = []
        Y = []

        for it, (x, y) in enumerate(loader):
            X.append(x)
            Y.append(y.item())

        X = torch.cat(X)
        scores = []
        for l, layer in enumerate(self.layers):
            #print('Supervising Layer ', l)
            layer.supervise(X, Y, task, iter)
            score, score_pc = layer.get_ci()
            scores.append(score)

            plt.bar(np.arange(len(score_pc)), score_pc)
            plt.xlabel('Classes')
            plt.ylabel('Counts')
            plt.savefig(smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/iter_{iter}') + f'layer_L{l}_cin_pc.png')
            plt.close()

        plt.bar(np.arange(len(scores)), scores)
        plt.xlabel('Layers')
        plt.ylabel('CIN %')
        plt.savefig(smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/iter_{iter}') + 'cin.png')

        

    # call classification function
    def classify(self, loader, task, iter):
        X = []
        Y = []
        for it, (x, y) in enumerate(loader):
            X.append(x)
            Y.append(y.item())

        X = torch.cat(X)
        y_true = np.array(Y)
        num_classes = len(np.unique(y_true))
        votes = []
        layer_score = []

        for l, layer in enumerate(self.layers):
            v_l, _ = layer.classify(X, Y, task, iter)
            votes.append(v_l)

            y_pred = torch.argmax(v_l, dim=1).detach().cpu().numpy()
            
            
            layer_score.append(100 * np.mean(y_true == y_pred))

            # Visualization
            confusion_matrix = np.zeros((num_classes, num_classes))

            for cf in range(len(y_pred)):
                confusion_matrix[int(y_true[cf]), int(y_pred[cf])] += 1

            plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
            plt.ylabel("True Class")
            plt.xlabel("Predicted Class")
            plt.xticks(np.arange(num_classes))
            plt.yticks(np.arange(num_classes))
            for i in range(num_classes):
                for j in range(num_classes):
                    text = plt.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="w")

            plt.savefig(smart_dir(f"logs/{self.log}/{self.dataset}/" + f'task_{task}/iter_{iter}/layer_L{l}') + 'class_confusion_matrix.png')
            plt.close()
            np.save(smart_dir(f"logs/{self.log}/{self.dataset}/" + f'task_{task}/iter_{iter}/layer_L{l}') + 'class_confusion_matrix.npz', confusion_matrix)
            
        all_votes = torch.stack(votes)
        y_pred = torch.argmax(torch.sum(all_votes, dim=0), dim=1).detach().cpu().numpy()

        
        confusion_matrix = np.zeros((num_classes, num_classes))

        for cf in range(len(y_pred)):
            confusion_matrix[int(y_true[cf]), int(y_pred[cf])] += 1

        plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.xticks(np.arange(num_classes))
        plt.yticks(np.arange(num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                text = plt.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="w")

        plt.savefig(smart_dir(f"logs/{self.log}/{self.dataset}/" + f'task_{task}/iter_{iter}/') + 'class_confusion_matrix.png')
        plt.close()

        ### Visualize layer-wise performance
        plt.plot(layer_score)
        plt.title('Classwise Class Performance')
        plt.xlabel('Layers')
        plt.ylabel('Performance')
        plt.savefig(smart_dir(f"logs/{self.log}/{self.dataset}/" + f'task_{task}/iter_{iter}') + 'layer_class_acc.png')
        plt.close()

        
        score = 100 * np.mean(y_true == y_pred)

        score_pc = [100*confusion_matrix[i, i] / np.sum(confusion_matrix[i]) for i in range(num_classes)]

        return score, score_pc

       
    def jaccard(self, x, y):
        x = set(x)
        y = set(y)
        val = len(x.intersection(y)) / len(x.union(y))
        
        if val == None:
            return 0
        else:
            return val
        
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
    #@profile
    def cluster(self, loader, task, iter, k_scale=2):

        zs = []
        ys = []
        for it, (data, y) in enumerate(loader):
            embed = []
            ys.append(y.item())
            for layer in self.layers:
                z = layer.embed(data)
                embed.append(z)
            zs.append(np.concatenate(embed))
        
        zs = np.stack(zs)
        y_true = np.array(ys)
        num_classes = len(np.unique(ys))

        similarity_matrix = pairwise_distances(zs, zs, metric=self.jaccard)
        
        k = num_classes * k_scale
        accu_total = 0
        accu_perclass = np.zeros(num_classes, dtype=np.float64)

        y_pred = SpectralClustering(n_clusters=k, affinity='precomputed', n_init=10,
                                            assign_labels='discretize').fit_predict(similarity_matrix)

        
        accu_total, accu_perclass = self.purity_score(y_true, y_pred)

        confusion_matrix = np.zeros((num_classes, k))

        for i in range(len(y_pred)):
            confusion_matrix[int(y_true[i]), int(y_pred[i])] += 1

        plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.xticks(np.arange(k))
        plt.yticks(np.arange(num_classes))
        for i in range(num_classes):
            for j in range(k):
                text = plt.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="w")

        plt.savefig(smart_dir(f"logs/{self.log}/{self.dataset}/" + f'task_{task}/iter_{iter}') + 'cluster_confusion_matrix.png')
        plt.close()

       #self.model_stats['clust_acc'].append(accu_total)

        return accu_total*100, accu_perclass


    def eval(self, super_loader, test_loader, t, it):
        #print('Supervising...')
        self.supervise(super_loader, task=t, iter=it)

        #print('Classifying...')
        class_acc, class_pc_acc = self.classify(test_loader, task=t, iter=it)

        print('Classification Accuracy: ', class_acc)
        #print('PC Accuracy: ', class_pc_acc)

        #print('Clustering...')
        clust_acc, clust_pc_acc = self.cluster(test_loader, task=t, iter=it)


        print('Clustering Purity: ', clust_acc)
        #print('PC Purity: ', clust_pc_acc)

        self.plots(t, it)

        return class_acc, class_pc_acc, clust_acc, clust_pc_acc
    
    def plots(self, t, it):

        for l, layer in enumerate(self.layers):
            plt.plot(layer.ltm_size_history, label=f'Layer_{l}')
            np.save(smart_dir(f"logs/{self.log}/{self.dataset}/" + f'task_{t}/iter_{it}/layer_L{l}') + 'ltm_size_history.npy', layer.ltm_size_history)
            
        plt.savefig(smart_dir(f"logs/{self.log}/{self.dataset}/" + f'task_{t}/iter_{it}') + 'ltm_size.png')
        plt.close()

        for l, layer in enumerate(self.layers):
            plt.plot(layer.distance_threshold_history, label=f'Layer_{l}')
            
        plt.savefig(smart_dir(f"logs/{self.log}/{self.dataset}/" + f'task_{t}/iter_{it}') + 'nd_thresh_history.png')
        plt.close()





