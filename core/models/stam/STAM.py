import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.models.stam.stam_layer import Layer
from core.utils import *

from sklearn.cluster import KMeans, SpectralClustering
#from kmodes.kmodes import KModes
from sklearn.metrics import pairwise_distances
from torchmetrics.functional import pairwise_euclidean_distance



class STAM():

    def __init__(self, config):
        
        # declare properties
        self.name = 'STAM'

        # extract scenario configs
        self.im_size = config.dataset.img_size
        self.num_c = config.dataset.channels
        self.seed = config.seed

        # extract stam configs
        self.num_layers = config.model.num_layers

        # build stam hierarchy
        self.layers = []
        for l in range(self.num_layers):
            self.layers.append(Layer(l, config))

        self.rho = config.model.rho
        # stam init
        self.initialized = False
        self.task = 0


        # Logging
        self.dataset = config.dataset.name
        self.log = config.log


        # stam init
        self.init_layers()



    # centroid init - note that current implementation does NOT use random 
    # init centroids but rather will change these centroids to sampled patch 
    # values in the learning alogrithm (see STAM_classRepo)
    def init_layers(self):
        
        # random seed
        np.random.seed(self.seed)

    def pretrain(self, loader):

        for it, batch in tqdm(enumerate(loader), total=len(loader)):
            x, y = batch
            x = x.numpy().transpose(0, 2, 3, 1)
            
            for i in range(len(x)):
                for l in range(self.num_layers):
                    self.layers[l](x[0])

        print('Done with T0')


    def __call__(self, x, t):
            x = x.numpy().transpose(0, 2, 3, 1)

            for l in range(self.num_layers):
                self.layers[l](x[0])


    # get percent class informative centroids
    def get_ci(self, num_classes):

        # hold results here
        score = [0 for l in range(self.num_layers)]
        score_pc = [np.zeros((num_classes,)) for l in range(self.num_layers)]

        # for each layer
        for l in range(self.num_layers):

            # for each centroid
            for j in range(len(self.cent_g[l])):

                # increase score if ci
                if max(self.cent_g[l][j]) > self.rho_task:
                    score[l] += 1

                for k in range(num_classes):
                    if self.cent_g[l][j,k] > self.rho_task:
                        score_pc[l][k] += 1
            
            # calculate percent ci at layer
            score[l] /= len(self.cent_g[l])
            score_pc[l] /= len(self.cent_g[l])

        
        return np.asarray(score), np.asarray(score_pc)

    # given labeled x, associate class information with stam centroids
    #@profile
    def supervise(self, loader, task):
        x = []
        labels = []
        for i, (data, y) in enumerate(loader):
            x.append(data.numpy().transpose(0, 2, 3, 1))
            labels += y.numpy().tolist()

        x = np.concatenate(x)
        l_list = np.arange(len(self.layers))

        # process inputs
        num_x = len(x)
        num_classes = len(np.unique(labels))
        num_labels = len(np.unique(labels))
        self.rho_task = (1/num_labels) + self.rho

        # get centroids for classification
        self.cents_ltm = []

        for l in range(self.num_layers):
            if len(self.layers[l].centroids) > self.layers[l].delta:
                self.cents_ltm.append(self.layers[l].get_ltm_centroids())
            else:
                self.cents_ltm.append(self.layers[l].get_stm_centroids())


        self.cent_g = []

        # supervision per layer
        for l_index in range(len(l_list)):

            # get layer index from list of classification layers
            l = l_list[l_index]
        
            # get layer centroids
            centroids = self.layers[l_index].get_ltm_centroids()
            num_centroids = int(len(centroids))

            D = 0
            
            # get value of D for task
            # we use D to normalize distances wrt average centroid-patch distance
            for i in range(num_x):
            
                # get input to layer l
                x_i = x[i]

                # extract patches
                xp = self.layers[l].extract_patches(x_i)
                xp = xp.reshape(xp.shape[0], -1)
                
                # calculate and save distance
                cp = centroids.reshape(num_centroids, -1)
                d = smart_dist(xp, cp)
                close_ind = np.argmin(d, axis = 1)
                D += np.sum(d[range(xp.shape[0]),close_ind]) / xp.shape[0]

            # final D calculation    
            D = D / num_x
                       
            # this holds sum of exponential "score" for each centroid for each class
            sum_fz_pool = np.zeros((num_centroids, num_classes))


            # for each image
            for i in range(num_x):
            
                # get input to layer l
                x_i = x[i]
            
                # extract patches
                xp = self.layers[l].extract_patches(x_i)
                xp = xp.reshape(xp.shape[0], -1)

                # calculate distance
                cp = centroids.reshape(num_centroids, -1)
                d = smart_dist(xp, cp)

                # get distance of *matched* centroid of each patch
                close_ind = np.argmin(d, axis = 1)
                dist = (d[range(xp.shape[0]),close_ind])

                # get exponential distance and put into sparse array with same shape as 
                # summed exponential scores if we have two centroid matches in same 
                # image, only save best match
                td = np.zeros(d.shape)
                td[range(xp.shape[0]),close_ind] = np.exp(-1*dist/D)
                fz = np.amax(td, axis = 0)
                
                # update sum of exponential "score" for each centroid for each class
                sum_fz_pool[:, int(labels[i])] += fz

            # save x scores and calculate g values as exponential "score" normalized 
            # accross classes (i.e. score of each centroid sums to 1)
            Fz = sum_fz_pool    
            self.cent_g.append(np.copy(sum_fz_pool))

            for j in range(num_centroids):
                self.cent_g[l_index][j,:] = self.cent_g[l_index][j,:] \
                    / (np.sum(self.cent_g[l_index][j,:]) + 1e-5)


    def eval(self, sup_loader, eval_loader, task, it=None):

        print('Supervising...')
        self.supervise(sup_loader, task)

        print('Classifying...')
        class_acc, class_acc_pc = self.classify(eval_loader, task)

        print('Clustering...')
        clust_acc, clust_acc_pc = self.cluster(eval_loader, task)

        return class_acc, class_acc_pc, clust_acc, clust_acc_pc

    # call classification function
    def classify(self, loader, task):
        x = []
        labels_true = []
        for i, (data, y) in enumerate(loader):
            x.append(data.numpy().transpose(0, 2, 3, 1))
            labels_true += y.numpy().tolist()

        x = np.concatenate(x)
        num_classes = len(np.unique(labels_true))
        labels_true = np.array(labels_true)

        labels_predict = self.topDownClassify(x, num_classes)

        acc = np.sum(np.array(labels_true) == np.array(labels_predict)) / len(labels_true)

        acc_pc = np.zeros(num_classes)
        for y in range(num_classes):
            inds_y = np.argwhere(labels_true == y).flatten()
            acc_pc[y] = np.sum(np.array(labels_true[inds_y]) == np.array(labels_predict[inds_y])) / len(labels_true[inds_y])
        
        return acc * 100, acc_pc * 100
    
    # stam primary classification function - hierarchical voting mechanism
    #@profile
    def topDownClassify(self, x, num_classes):

        # process inputs and init return labels
        num_x = len(x)
        labels = -1 * np.ones((num_x,))

        # for each x
        for i in range(num_x):

            # get NN centroid for each patch
            close_ind = []
            close_distances = []
            for l in range(self.num_layers):

                # get ltm centroids at layer
                centroids = self.cents_ltm[l]
                num_centroids = int(len(centroids))

                # get input to layer
                x_i = x[i]

                # extract patches
                xp = self.layers[l].extract_patches(x_i)
                xp = xp.reshape(xp.shape[0], -1)


                # calculate distance
                cp = centroids.reshape(num_centroids, -1)
                d = smart_dist(xp, cp)
                close_ind.append(np.argmin(d, axis = 1))
                close_distances.append(np.min(d, axis = 1))
            
            
            # get highest layer containing at least one CIN centroid
            l = self.num_layers-1
            found_cin = False
            while l > 0 and not found_cin:
            
                # is there at least one CIN centroid?
                if np.amax(self.cent_g[l][close_ind[l]]) >= self.rho_task:
                    found_cin = True
                else:
                    l -= 1
            l_cin = l
                        
            # classification
            #
            # vote of each class for all layers
            wta_total = np.zeros((num_classes,)) + 1e-3

            # for all cin layers
            layer_range = range(l_cin+1)
            percent_inform = []
            for l in layer_range:
                # vote of each class in this layer
                wta = np.zeros((num_classes,))

                # get max g value for matched centroids
                votes_g = np.amax(self.cent_g[l][close_ind[l]], axis = 1)

                # nullify vote of non-cin centroids
                votes_g[votes_g < self.rho_task] = 0

                
                a = np.where(votes_g > self.rho_task)
                percent_inform.append(len(a[0])/ len(votes_g))      

                # calculate per class vote at this layer
                votes = np.argmax(self.cent_g[l][close_ind[l]], axis = 1)
                for k in range(num_classes):
                    wta[k] = np.sum(votes_g[votes == k])

                # add to cumalitive total and normalize
                wta /= len(close_ind[l])
                
                wta_total += wta

            # final step
            labels[i] = np.argmax(wta_total)

        return labels

    #@profile
    def embed_patch(self, X, layer, cents_ltm):

        # patch size and num_features calculations
        p = layer.patch_size
        n_cols = len(layer.extract_patches(X[0]))

        X_ = np.zeros((X.shape[0], n_cols), dtype=int)
        

        for i, x in enumerate(X):

            # extract patches
            patches = layer.extract_patches(x)
            patches = patches.reshape(patches.shape[0], -1)

            d_mat = smart_dist(patches, cents_ltm)

            # get indices of closest patch to each centroid and accumulate average 
            # closest-patch distances
            close_patch_inds = np.argmin(d_mat, axis=1)
            X_[i] = close_patch_inds
        
        print("Got Jaccard Embedding")
        return X_

    def jaccard(self, x, y):
        x = set(x)
        y = set(y)
        val = len(x.intersection(y)) / len(x.union(y))
        
        if val == None:
            return 0
        else:
            return val
    
    # cluster
    #@profile
    def cluster(self, loader, task):

        X = []
        Y = []
        for i, (data, y) in enumerate(loader):
            X.append(data.numpy().transpose(0, 2, 3, 1))
            Y += y.numpy().tolist()

        X = np.concatenate(X)
        Y = np.array(Y)
        num_classes = len(np.unique(Y))
        k_scale = 2
        # returns total and per-class accuracy... (float, 1 by k numpy[float])
        print('Clustering Task Started...')

        similarity_matrix = np.zeros(10)

        embeddings = self.embed_patch(X, self.layers[-1], self.cents_ltm[-1])

        similarity_matrix = pairwise_distances(embeddings, embeddings, metric=self.jaccard)
        
        k = num_classes * k_scale
        accu_total = 0
        accu_perclass = np.zeros(num_classes, dtype=np.float64)

        # Clustering Predictions

        try:
            cluster_preds = SpectralClustering(n_clusters=k, affinity='precomputed', n_init=10,
                                            assign_labels='discretize').fit_predict(similarity_matrix)
        except Exception as e:
            cluster_preds = np.zeros(len(similarity_matrix))

        # Accuracy of Clustering

        size = k
        cluster_counts = np.zeros((size, int(k/k_scale)))
        cluster_sizes = np.zeros(size)
        correct = np.zeros(size)
        total = np.zeros(size)
        cluster_indicies = [[] for i in range(num_classes)]

        for i in range(size):
            cluster_i = np.argwhere(cluster_preds == i).flatten()  # indexes of cluster i
            cluster_sizes[i] = len(cluster_i)
            cluster_counts[i,:] = np.bincount(Y[cluster_i], minlength=int(k/k_scale))

            # compute accuracy
            cluster_class = np.argmax(cluster_counts[i, :])
            correct[i] = cluster_counts[i, :].max()
            total[i] = cluster_counts[i, :].sum()
            cluster_indicies[cluster_class].append(i)

        for j in range(num_classes):
            if sum(total[cluster_indicies[j]]) > 0:
                accu_perclass[j] = sum(correct[cluster_indicies[j]]) \
                    / sum(total[cluster_indicies[j]]) * 100
            else:
                accu_perclass[j] = 0

        accu_total = sum(correct) / sum(total) * 100
    

        return accu_total, accu_perclass

    # save STAM visualizations    
    def save_visualizations(self, save_dir, task):
        
        # Cent count
        plt.figure(figsize=(6,3)) 
        for l in range(self.num_layers):
            y = np.asarray(self.layers[l].ltm_size_history)
            x = np.arange(len(y))
            plt.plot(x, y, label = 'layer ' + str(l+1))
        plt.ylabel('LTM Count', fontsize=12)
        plt.xlabel('Unlabeled Images Seen', fontsize=12) 
        plt.title('LTM Centroid Count History', fontsize=14)
        plt.legend(loc='upper left', prop={'size': 8})
        plt.grid()
        plt.tight_layout()
        plt.savefig(smart_dir(save_dir+'cent_plots')+'ltm_count.png', format='png', dpi=200)
        plt.close()

        
        #for l in range(self.num_layers):
        #    np.savetxt(smart_dir(save_dir+'ltm_csvs') + 'layer-' + str(l+1) + '_ci.csv', 
        #               self.layers[l].ltm_size_history, delimiter=',')

        # confidence interval
        plt.figure(figsize=(6,3))  
        p = np.asarray([0, 25, 50, 75, 90, 100])
        for l in range(self.num_layers):
            dd = np.asarray(np.sort(self.layers[l].window))            
            y = np.percentile(dd, p)
            x = np.arange(len(dd)) / len(dd)
            plt.plot(x, dd, label = 'layer ' + str(l+1))
            plt.plot(p/100., y, 'ro')
            plt.axhline(y=self.layers[l].distance_threshold, color='r', linestyle='--')
        plt.xticks(p/100., map(str, p))
        plt.ylabel('Distance', fontsize=12)
        plt.xlabel('Percentile', fontsize=12)
        plt.title('Distribution of Closest Matching Distance', fontsize=14)
        plt.legend(loc='lower right', prop={'size': 8})
        plt.grid()
        plt.tight_layout()
        plt.savefig(smart_dir(save_dir+'cent_plots')+'d-thresh.png', format='png', dpi=200)
        plt.grid()
        plt.close()

        # D threshold
        plt.figure(figsize=(6,3)) 
        for l in range(self.num_layers):
            y = np.asarray(self.layers[l].distance_threshold_history)
            x = np.arange(len(y))
            plt.plot(x, y, label = 'layer ' + str(l+1))
        plt.ylabel('ND Distance', fontsize=12)
        plt.xlabel('Unlabeled Images Seen', fontsize=12)  
        plt.gca().set_ylim(bottom=0)
        plt.title('Novelty Detection Threshold History', fontsize=14)
        plt.legend(loc='upper left', prop={'size': 8})
        plt.grid()
        plt.tight_layout()
        plt.savefig(smart_dir(save_dir+'cent_plots')+'d-thresh-history.png', 
                    format='png', dpi=200)
        plt.close()

    def plots(self):
        return
        
    
    def detailed_classification_plots(self, save_dir):
        if self.out_layer != 3:
            return
        index = 0
        labels = -1 * np.ones((len(self.sample_images),))
        
        for i in range(len(self.sample_images)):

            close_ind = []
            for l in range(self.num_layers):

                # get ltm centroids at layer
                centroids = self.cents_ltm[l]
                num_centroids = int(len(centroids))

                # get input to layer
                x_i = self.sample_images[i]
                #for l_ in range(l): 
                #    x_i = self.layers[l_](x_i)

                # extract patches
                xp = self.layers[l].extract_patches(x_i)
                xp = xp.reshape(xp.shape[0], -1)

                # calculate distance
                cp = centroids.reshape(num_centroids, -1)
                d = smart_dist(xp, cp)
                close_ind.append(np.argmin(d, axis = 1))
            
            
            # get highest layer containing at least one CIN centroid
            l = self.num_layers-1
            found_cin = False
            while l > 0 and not found_cin:
            
                # is there at least one CIN centroid?
                if np.amax(self.cent_g[index][l][close_ind[l]]) >= self.rho_task:
                    found_cin = True
                else:
                    l -= 1
            l_cin = l
                        
            # classification
            #
            # vote of each class for all layers
            wta_total = np.zeros((self.num_classes,)) + 1e-3

            # for all cin layers
            layer_range = range(3)
            percent_inform = []
            layer_wta = []
            layer_vote_counts = []
            for l in layer_range:
                # vote of each class in this layer
                wta = np.zeros((self.num_classes,))

                # get max g value for matched centroids
                votes_g = np.amax(self.cent_g[index][l][close_ind[l]], axis = 1)

                # nullify vote of non-cin centroids
                votes_g[votes_g < self.rho_task] = 0
                a = np.where(votes_g > self.rho_task)
                percent_inform.append(len(a)/ len(votes_g))      

                # calculate per class vote at this layer
                votes = np.argmax(self.cent_g[index][l][close_ind[l]], axis = 1)
                layer_vote_counts.append(np.bincount(votes[a], minlength=self.num_classes))

                for k in range(self.num_classes):
                    wta[k] = np.sum(votes_g[votes == k])

                # add to cumalitive total
                layer_wta.append(wta)
                wta /= len(close_ind[l])
                wta_total += wta
                    
            # final step
            labels[i] = np.argmax(wta_total)
            
            # Visualizing Patches and Centroids
            for l in range(self.num_layers):
                nrows = ncols = int(np.sqrt(self.layers[l].num_patches) / 2)
                rf_size = self.layers[l].patch_size
                
                plt.close()
                fig = plt.figure(figsize=(9,11))

                # First 3
                out_im, out_im_2 = self.layers[l].create_reconstruction(self.sample_images[i], self.sample_labels[i])
                ax1 = fig.add_axes([0.1, 0.75, 0.2, 0.2])

                ax2 = fig.add_axes([0.35, 0.75, 0.2, 0.2])

                ax3 = fig.add_axes([0.63, 0.83, 0.30, 0.12])

                ax1.imshow(out_im_2.squeeze())
                ax1.set_title('Patches')
                ax1.axis('off')
                
                ax2.imshow(out_im.squeeze())
                ax2.set_title('Matched Centroids')
                ax2.axis('off')
                
                ax3.bar(np.arange(self.num_classes), layer_vote_counts[l])
                ax3.set_xticks(np.arange(self.num_classes))
                ax3.set_xticklabels(self.class_labels, rotation='vertical')
                ax2.tick_params(axis='y', which='major', labelsize=10)
                ax3.set_title('Layer {} Vote  (1/K + Gamma): {}'.format(l, self.rho_task))
                ax3.axis('on')
                
                patches = self.layers[l].extract_patches(x_i)
                patches = patches.reshape(patches.shape[0], -1)
                xp_2 = patches.reshape(self.layers[l].num_patches, -1)

                centroids = self.cents_ltm[l]
                num_centroids = int(len(centroids))
                cp = centroids.reshape(num_centroids, -1)

                for p in range(4):
                    for j in range(5):
                        if int((p*ncols*2) + (2*j)) >= len(xp_2):
                            continue
                        ax1 = fig.add_axes([0.08 + .17*j, 0.57 - .15*p, 0.05, 0.05])
                        ax2 = fig.add_axes([0.16 + .17*j, 0.57 - .15*p, 0.05, 0.05])
                        ax3 = fig.add_axes([0.08 + .17*j, 0.65 - .15*p, .13, .05])

                        ax1.set_title('Patch')
                        pat = xp_2[int((p*ncols*2) + (2*j))].reshape(rf_size, rf_size, self.num_c).squeeze() - xp_2[int((p*ncols*2) + (2*j))].min()
                        pat /= pat.max()
                        ax1.imshow(pat)
                        ax1.axis('off')

                        if np.max(self.cent_g[index][l][close_ind[l][int((p*ncols*2) + (j*2))]]) > self.rho_task:
                            ax2.set_title('Centroid', color='g')
                        else:
                            ax2.set_title('Centroid', color='r')
                        pat = cp[close_ind[l][int((p*ncols*2) + (j*2))]].reshape(rf_size, rf_size, self.num_c).squeeze() - cp[close_ind[l][int((p*ncols*2) + (j*2))]].min()
                        pat /= pat.max()
                        ax2.imshow(pat)
                        ax2.axis('off')
                        
                        vote = np.argmax(self.cent_g[index][l][close_ind[l][int((p*ncols*2) + (j*2))]])
                        ax3.set_title('Vote: {}'.format(self.class_labels[vote]))
                        ax3.bar(np.arange(self.num_classes), self.cent_g[index][l][close_ind[l][int((p*ncols*2) + (j*2))]])
                        ax3.axes.get_xaxis().set_ticks([])
                        ax3.tick_params(axis='y', which='major', labelsize=6)
                        #ax3.axis('off')
                    

                fig.suptitle('True Class: {}   Predicted Class: {}   Layer{}'.format(self.class_labels[int(self.sample_labels[i])], self.class_labels[int(labels[i])], l))
                plt.savefig(smart_dir(save_dir + 'task_{}/ex_{}/'.format(self.task, i)) + 'layer_{}_vote.png'.format(l))    
                plt.close()

        return labels


