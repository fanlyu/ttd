import numpy as np
import math

from numpy.lib import stride_tricks
from sklearn.utils import resample
from collections import deque
from core.utils import *
import matplotlib.pyplot as pltß
import torchvision.utils as vutils

import torchvision.transforms as T


def perc(array, percent):

    k = (len(array)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return array[int(k)]
    d0 = array[int(f)] * (c-k)
    d1 = array[int(c)] * (k-f)
    return d0+d1


# The STAM Layer Class
class Layer:

    def __init__(self, l, config):

        # General Layer Parameters
        params = config.model.layers[f'layer{l}']  
        self.name = params.name                                              # name of layer
        self.ch = params.ch                                                  # num channels
        self.patch_size = params.patch_size                                  # size of rf i.e. patch
        self.img_size = config.dataset.img_size
        self.stride = params.stride                                          # stam stride for clustering
        self.alpha = params.alpha                                            # centroid learning rate (stm)
        self.ltm_alpha = params.ltm_alpha                                    # centroid learning rate (ltm) - often zero
        self.beta = params.beta                                              # percentile of distance distribution for novelty detection
        self.theta = params.theta                                            # stm activations required for ltm
        self.delta = int(params.delta)                                       # stm size
        self.normalize = params.normalize                                    # Normalize values for dataset images                                                 # The memory size for LTM
        self.rho_task = params.rho                             

        # Patch Extraction Parameters
        self.num_patches = int(np.power(np.floor((self.img_size - self.patch_size)    # calculate number of receptive fields total at cluster level
            / self.stride) + 1, 2))
        self.num_patch_axis = int(np.ceil((self.img_size - self.patch_size)       # calculate RFs per axis at reconstruction level
            / 1) + 1)

        # init param
        self.num_init = 0
        self.init_samples = 10
        self.step = 0

        # The centroids
        self.centroids = np.zeros((self.delta, self.patch_size**2 * self.ch))
        # holds the most recent time centroid has been active
        self.stm_unused_count = np.zeros((self.delta,),float)
        # hold centroid stm_count
        self.stm_matches = np.zeros((self.centroids.shape[0],),float)


        # holds centroid statistics
        self.distance_threshold = -1 # novelty detection threshold

        # Novelty Detection Variables
        self.window_size = 100
        self.window = deque(maxlen=self.window_size)
        self.num_distance_samples = 10

        # Visualization Stats
        self.distance_threshold_history = []
        self.ltm_size_history = []

     # scale patches if flag set
    def scale(self, x):
        shift = np.mean(x, axis = 1)[:,None]
        x -= shift
        scale = np.std(x, axis = 1)[:,None]
        scale[scale == 0] = 1
        x_out = x / scale
        return x_out

    # get ALL patches from image    
    def extract_patches(self, im):
        shape = (self.num_patch_axis, self.num_patch_axis, self.patch_size, self.patch_size, self.ch)
        strides = (im.strides[0], im.strides[1], im.strides[0], im.strides[1], im.strides[2])
        patches = stride_tricks.as_strided(im, shape=shape, strides=strides)
        patches = patches[range(0,self.num_patch_axis,self.stride),:][:,range(0,self.num_patch_axis,self.stride),:,:,:]

        return self.scale(patches.reshape((self.num_patches, -1)))


    def __call__(self, x):
        # STM is not full add to initial STM, bootstrap if STM full
        if self.num_init < self.delta:
            self.add_init_samples(x)

        # STM is full - boostrap novel distance threshold
            if self.num_init >= self.delta:
                print('Boostrapping Novelty')
                self.boostrap_novelty()
                print('Done')

            return x

        # STM is initialized - normal forward pass
        else:
            # Extract Patches
            patches = self.extract_patches(x)

            # Pairwise distance between patches and centroids
            distances = smart_dist(patches, self.centroids)

            # Distance to closest centroid for each patch
            patch_cent_dist = np.amin(distances, axis = 1)
            
            # Indices of closest centroid for each patch
            patch_cent_ind = np.argmin(distances, axis = 1)
            
            # Indices of closest patch for each centroid
            cent_patch_ind = np.argmin(distances, axis = 0)

            #####################
            ## STM REPLACEMENT ##
            #####################
            
            # get indexes of patches with a novel distance to closest centroid
            novel_indices = np.argwhere(patch_cent_dist > self.distance_threshold).flatten()
            # Number of these patches     
            num_novelties = len(novel_indices)

            if num_novelties > 0:

                # Copy stm_unused_count to do some extra filtering
                nd_centroid_recency = np.copy(self.stm_unused_count)
                
                # Set matched centroids to -1 (smaller than all possible values)
                nd_centroid_recency[patch_cent_ind[patch_cent_ind < self.delta]] = -1

                # Get num_novelties least recently used indices
                forget_inds = np.argsort(nd_centroid_recency)[-num_novelties:]

                # New Addition in case number of novelties is greater than delta
                if num_novelties > self.delta:
                    self.centroids[np.arange(self.delta)] = patches[novel_indices[np.arange(self.delta)]]
                    self.stm_unused_count[forget_inds] = 0
                    self.stm_matches[forget_inds] = 0

                    patch_cent_ind[np.arange(self.delta)] = forget_inds[np.arange(self.delta)]
                # This is the original STAM version
                else:

                    # Replace STM centroids with new novel patches
                    self.centroids[forget_inds] = patches[novel_indices]
                    self.stm_unused_count[forget_inds] = 0
                    self.stm_matches[forget_inds] = 0

                    # ????
                    patch_cent_ind[novel_indices] = forget_inds

            #####################
            ## Centroid Update ##
            #####################

            # Do an update for each patch - filtering below
            patch_update = [True for i in range(len(patches))]
            for i in range(len(patches)):

                # Distances to all patches from the chosen centroid
                ndist = distances[:,patch_cent_ind[i]]

                # If this patch is not the closest set update to false
                if not np.argmin(ndist) == i:
                    patch_update[i] = False
            
            # All available cent indices ready for update
            cents_matched = patch_cent_ind[patch_update]

            # Get the cents in STM
            cents_matched_stm = cents_matched[cents_matched < self.delta]
            # Remove STM cents already copied to LTM (stm_matches == -1)
            cents_matched_stm = cents_matched_stm[self.stm_matches[cents_matched_stm] >= 0]

            # Get the cents in LTM
            cents_matched_ltm = cents_matched[cents_matched >= self.delta]

            # If we have at least one matched stm cent
            if len(cents_matched_stm) > 0:
                # Indices of patches that matched to STM centroids
                patches_matched_stm = cent_patch_ind[cents_matched_stm]

                # Incremental STM centroid Update
                self.centroids[cents_matched_stm] = (1 - self.alpha) * self.centroids[cents_matched_stm] \
                    + self.alpha * patches[patches_matched_stm]

            # If we have at least one matched ltm cent
            if len(cents_matched_ltm) > 0:
                # Indices of patches that matched to LTM centroids
                patches_matched_ltm = cent_patch_ind[cents_matched_ltm]

                # Incremental LTM centroid update (ltm_alpha currently 0 so does nothing)
                self.centroids[cents_matched_ltm] = (1 - self.ltm_alpha) * self.centroids[cents_matched_ltm] \
                    + self.ltm_alpha * patches[patches_matched_ltm]

            # Indices of centroids that were updated
            update_index = np.unique(patch_cent_ind) 

            # Get just the STM cents
            update_index = update_index[update_index < self.delta]
            # Remove cents already copied to LTM (stm_matches == -1)
            self.stm_matches[update_index[self.stm_matches[update_index] >= 0]] += 1
            # Increment all STM centroids as unused once
            self.stm_unused_count += 1
            # Reset unused counter to zero for matched STM centroids
            self.stm_unused_count[update_index[update_index < self.delta]] = 0
            
            #######################
            ## Novelty Detection ##
            #######################

            # Sample num_distance_samples distances to add to sliding window
            sampled_d = np.random.choice(patch_cent_dist, self.num_distance_samples)
            # Add samples to sliding window
            self.window.extend(sampled_d)

            # Update Distance Threshold
            self.distance_threshold = perc(np.sort(np.array(self.window)), self.beta)

            ###################
            ## LTM Additions ##
            ###################

            # Indices of mature STM centroids (stm_matches >= self.theta)
            mature_cent_inds = np.where(self.stm_matches >= self.theta)[0]

            # If there is at least one mature centroid
            if len(mature_cent_inds) > 0:
                # Set Existing STM Centroid to -1 to mark as not useable (can still be matched with but not updated)    
                self.stm_matches[mature_cent_inds] = -1
                
                # Create a new  LTM centroid
                self.centroids = np.append(self.centroids, np.copy(self.centroids[mature_cent_inds]), axis = 0)

                self.num_ltm = len(self.centroids) - self.delta
 
            # Visualization
            self.distance_threshold_history.append(self.distance_threshold)
            self.ltm_size_history.append(len(self.get_ltm_centroids()))

        self.step += 1

        return x

    def add_init_samples(self, x):
        # Extract patches from the image
        patches = self.extract_patches(x)
        
        # Insert init_sample number of patches into STM
        self.centroids[self.num_init:self.num_init+self.init_samples] = resample(patches, n_samples = self.init_samples)

        # Update the number of initialized centroids
        self.num_init += self.init_samples

        return x

    def boostrap_novelty(self):
        num_sample = min(self.delta, self.window_size)
        cent_samples = np.random.choice(self.delta, self.window_size)
        d_samples = []
        for j in cent_samples:
            d_samples.append(np.amin(smart_dist(self.centroids[j][None,:], self.centroids[np.arange(self.delta)!=j])))
        
        self.window.extend(d_samples)

        # Get distance threshold from beta percentile
        self.distance_threshold = perc(np.sort(np.array(self.window)), self.beta)
        
    # return ltm centroids
    def get_ltm_centroids(self):
        return self.centroids[self.delta:]

    # return stm centroids
    def get_stm_centroids(self):
        return self.centroids[:self.delta]
