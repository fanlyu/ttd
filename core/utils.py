import torch
import yaml, os, sys

#from neptune.utils import stringify_unsupported
from sklearn.metrics.pairwise import cosine_similarity
from pykeops.torch import LazyTensor
from numba import cuda

import numpy as np

from tqdm import tqdm

def l2_dist(x, y):
    
    xx = np.sum(x**2, axis = 1)
    yy = np.sum(y**2, axis = 1)
    xy = np.dot(x, y.transpose((1,0)))

    d = xx[:,None] - 2*xy + yy[None,:]
    d[d<0] = 0
    d = (d)**(1/2)
    
    return d

@cuda.jit
def euclidean_kernel(patches, centroids, out):
    """
    Euclidean Kernel for CUDA
    """
    p = patches.shape[0]
    c = centroids.shape[0]
    k = centroids.shape[1]
    x, y = cuda.grid(2)
    d = 0

    if x < p and y < c:
        for f in range(k):
            tmp = patches[x, f] - centroids[y, f]
            d += tmp * tmp
        out[x, y] = d ** 0.5
def cuda_wrapper(x, y, kernel):
    rows = x.shape[0]
    cols = y.shape[0]

    block_dim = (8, 8)

    grid_dim = (int(rows/block_dim[0] + 1), int(cols/block_dim[1] + 1))

    stream = cuda.stream()
    patches = cuda.to_device(x, stream=stream)
    centroids = cuda.to_device(y, stream=stream)
    out2 = cuda.device_array((rows, cols))
    kernel[grid_dim, block_dim](patches, centroids, out2)
    out = out2.copy_to_host(stream=stream)

    return out

def l1_dist(x, y):
    return np.sum(np.absolute(x[:,None,:]-y[None,:,:]), axis = 2)

def cosine_dist(x, y):
    return 1 - cosine_similarity(x, y)

def smart_dist(x, y, method="L2"):
    if method == "L1":
        return l1_dist(x, y)
    elif method == "L2":
        return l2_dist(x, y)
    elif method == "L2_CUDA":
        return cuda_wrapper(x, y, euclidean_kernel)
    elif method == "COS":
        return cosine_dist(x, y)
    else:
        return x


def KMeans_cosine(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Cosine similarity metric."""
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids
    # Normalize the centroids for the cosine similarity:
    c = torch.nn.functional.normalize(c, dim=1, p=2)

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in tqdm(range(Niter)):

        # E step: assign points to the closest cluster -------------------------
        S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
        cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Normalize the centroids, in place:
        c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

    return cl, c

# checks if directory exists; makes new directory if not exist
# returns directory name
def smart_dir(dir_name, base_list = None):
    dir_name = dir_name + '/'
    if base_list is None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name
    else:
        dir_names = []
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for d in range(len(base_list)):
            dir_names.append(dir_name + base_list[d] + '/')
            if not os.path.exists(dir_names[d]):
                os.makedirs(dir_names[d])
        return dir_names

'''
def load_config(run):
    with open("config/experiment.yaml", 'r') as f:
        experiment_config = yaml.safe_load(f)
        run["experiment_config"] = stringify_unsupported(experiment_config)

    with open(f"config/datasets/{experiment_config['dataset_config']}.yaml", 'r') as f:
        dataset_config = yaml.safe_load(f)
        run["dataset_config"] = stringify_unsupported(dataset_config)

    with open(f"config/models/{experiment_config['model_config']}.yaml", 'r') as f:
        model_config = yaml.safe_load(f)
        run["model_config"] = stringify_unsupported(model_config)

    experiment_config.update(dataset_config)
    experiment_config.update(model_config)

    return run, experiment_config
'''