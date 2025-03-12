import math, os, time

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
import torchvision.utils as vutils

from core.stream.collate import SleepCollateFunction, PatchCollateFunction
from core.stream.dataset import NumpyDataset
from core.models.pcmc.encoders import SimCLR, SwaV, SimSiam, BarlowTwins
from core.utils import smart_dir, KMeans_cosine
from core.stream.samplers import ExtendedSampler

from collections import deque

from torchmetrics.functional import pairwise_cosine_similarity
from lightly.data import LightlyDataset

from sklearn.cluster import AgglomerativeClustering, KMeans, AffinityPropagation
from sklearn.mixture import GaussianMixture

from tqdm import tqdm

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

        params = config.model.layers[f'layer{l}']
        # General Layer Parameters
        self.name = params.name                                              # name of layer
        self.ch = params.ch                                                  # num channels
        self.feat_size = params.feat_size
        self.patch_size = params.patch_size                                  # size of rf i.e. patch
        self.stride = params.stride                                          # stam stride for clustering
        self.alpha = params.alpha                                            # centroid learning rate (stm)
        self.ltm_alpha = params.ltm_alpha                                    # centroid learning rate (ltm) - often zero
        self.beta = params.beta                                              # percentile of distance distribution for novelty detection
        self.theta = params.theta                                            # stm activations required for ltm
        self.delta = int(params.delta)                                       # stm size
        self.mean = config.dataset.mean                                      # Normalize values for dataset images
        self.std = config.dataset.std
        self.rho_task = params.rho
        self.lr = params.lr
        self.wd = params.wd

        # init param
        self.num_init = 0
        self.init_samples = 10
        self.init_sample_factor = params.init_sample_factor

        # The Feature Model
        self.model = self.load_model(config, params)
        self.pretrain_only = params.pretrain_only
        self.color_jitter = T.ColorJitter(params.cj_b, params.cj_c, params.cj_s, params.cj_h)

        self.sleep_epochs = params.sleep_epochs
        self.init_epochs = params.init_epochs
        self.n_workers = config.model.n_workers
        self.sleep_bs = params.sleep_bs
        self.pretrain_bs = params.pretrain_bs

        self.sleep_on = config.model.sleep_on
        
        self.mem_update = config.model.mem_update
        self.update_use = config.model.update_use

        self.transform = [
            T.RandomHorizontalFlip(p=params.hf),
            #T.RandomVerticalFlip(p=params.vf),
            #T.RandomRotation(params.rot),
            T.RandomApply([self.color_jitter], p=params.cj),
            T.RandomGrayscale(p=params.gs),
            T.RandomApply([T.GaussianBlur(params.kn, (params.sigma1, params.sigma2))], params.gb),
            T.ToTensor(),
            #T.RandomErasing(params.re),
            T.Normalize(mean=self.mean, std=self.std),
        ]

        transform1 = T.Compose([T.Normalize(mean=[-self.mean[i] / self.std[i] for i in range(3)] , std=[1 / self.std[i] for i in range(3)]), T.ToPILImage(), 
                                T.RandomResizedCrop(size=self.patch_size, scale=(params.crop_min, params.crop_max))] 
                                + self.transform)
        transform2 = T.Compose([T.Normalize(mean=[-self.mean[i] / self.std[i] for i in range(3)] , std=[1 / self.std[i] for i in range(3)]), T.ToPILImage(), 
                                T.RandomResizedCrop(size=self.patch_size, scale=(1.0, 1.0))] 
                                + self.transform)


        self.sleep_collate_fn = SleepCollateFunction(self.patch_size, transform1, transform2)

        transform1 = T.Compose([T.RandomResizedCrop(size=self.patch_size, scale=(params.crop_min, params.crop_max))] + self.transform)
        transform2 = T.Compose([T.RandomResizedCrop(size=self.patch_size, scale=(1.0, 1.0))] + self.transform)

        self.pre_collate_fn = PatchCollateFunction(config.dataset.img_size, self.patch_size, transform1, transform2)

        self.stm = torch.zeros((self.delta, 512)).to('cuda')
        self.stm_matches = torch.zeros(self.delta)
        self.stm_unused_count = torch.zeros(self.delta)

        self.stm_ages = torch.zeros(self.delta)
        self.stm_examples = [[] for i in range(self.delta)]


        self.ltm = torch.zeros((0, 512)).to('cuda')
        self.ltm_task = torch.zeros(0)
        self.ltm_examples = []
        self.M = params.M
        self.forgetting_factor = params.forgetting_factor

        ## Centroid Consolidation
        self.sleep_freq = config.model.sleep_freq
        self.sleep_start = config.model.sleep_start
        self.init_clusters = params.init_clusters
        self.cluster_alg = config.model.cluster_alg

        ## Task Estimation
        self.sleep_cycles = 0
        self.nd_history = deque(maxlen=100) # maybe need a hyper-parameter

        # holds centroid statistics
        self.distance_threshold = -1 # novelty detection threshold

        # Novelty Detection Variables
        self.window_size = 1000
        self.window = deque(maxlen=self.window_size)
        self.num_distance_samples = 10

        # Visualization Stats
        self.distance_threshold_history = []
        self.ltm_size_history = []
        self.ltm_mem_size = []
        self.stm_mem_size = []
        self.sleep_mature = []

        # Logging
        self.pretrain_log = config.pretrain_log
        self.sleep_log = config.sleep_log
        self.load_pretrain = config.load_pretrain
        self.load_sleep = config.load_sleep
        self.dataset = config.dataset.name
        self.log = config.log
        self.plot = config.plot

    def load_model(self, config, params):
        if config.model.encoder_type == 'simclr':
            return SimCLR(params.temperature, config.model.pretrained).to('cuda')
        elif config.model.encoder_type == 'swav':
            return SwaV(params.temperature, params.init_clusters, config.model.pretrained).to('cuda')
        elif config.model.encoder_type == 'simsiam':
            return SimSiam(config.model.pretrained).to('cuda')
        elif config.model.encoder_type == 'barlow':
            return BarlowTwins(config.model.pretrained).to('cuda')

    def extract_patch_embeddings(self, x, return_raws=False):
        self.model.eval()
        self.model.backbone.eval()
        x = x.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
        x = x.permute(1, 2, 0, 3, 4)
        #x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.reshape(-1, 3, self.patch_size, self.patch_size)
        z = self.model.embed(x.to('cuda'))

        if return_raws:
            return z, x
        
        return z

    def get_ci(self):
        # hold results here
        score = 0
        score_pc = np.zeros(self.cent_g.shape[1])

        self.cin_inds = []
        # for each centroid
        for j in range(len(self.cent_g)):

            # increase score if ci
            if max(self.cent_g[j]) > self.rho_task:
                score += 1
                self.cin_inds.append(j)

            for k in range(self.cent_g.shape[1]):
                if self.cent_g[j,k] > self.rho_task:
                    score_pc[k] += 1
            
        # calculate percent ci at layer
        score /= len(self.cent_g)
        score_pc /= len(self.cent_g)

        
        return score, score_pc

    def supervise(self, X, Y, task, iter):
        self.num_centroids = int(len(self.ltm))

        # get value of D for task
        # we use D to normalize distances wrt average centroid-patch distance
        D_sum = 0
        for it, x in enumerate(X):
            z = self.extract_patch_embeddings(x)
            d = 1 - pairwise_cosine_similarity(z, self.ltm).detach().cpu().numpy()
            close_ind = np.argmin(d, axis = 1)
            D_sum += np.sum(d[range(z.shape[0]),close_ind]) / z.shape[0]
            

        # final D calculation    
        D = D_sum / len(X)

 
        # this holds sum of exponential "score" for each centroid for each class
        sum_fz_pool = np.zeros((self.num_centroids, len(np.unique(Y))))

        supervise_matches = [[] for i in range(self.num_centroids)]
        # for each image
        for it, x in enumerate(X):
            z, p = self.extract_patch_embeddings(x, return_raws=True)
            d = 1 - pairwise_cosine_similarity(z, self.ltm).detach().cpu().numpy()
            # get distance of *matched* centroid of each patch
            close_ind = np.argmin(d, axis = 1)
            dist = (d[range(z.shape[0]),close_ind])


            ## Visualizing the matched centroids
            for j, ind in enumerate(close_ind):
                supervise_matches[ind].append(p[j])

            # get exponential distance and put into sparse array with same shape as 
            # summed exponential scores if we have two centroid matches in same 
            # image, only save best match
            td = np.zeros(d.shape)
            td[range(z.shape[0]),close_ind] = np.exp(-1*dist/D)
            fz = np.amax(td, axis = 0)
            
            # update sum of exponential "score" for each centroid for each class
            sum_fz_pool[:, int(Y[it])] += fz

        # save x scores and calculate g values as exponential "score" normalized 
        # accross classes (i.e. score of each centroid sums to 1)
        #Fz = sum_fz_pool    
        self.cent_g = np.copy(sum_fz_pool)

        for j in range(self.num_centroids):
            self.cent_g[j,:] = self.cent_g[j,:] / (np.sum(self.cent_g[j,:]) + 1e-5)

        ### Visualize
        if self.plot:
            supervise_matches = [torch.stack(m) if len(m) > 0 else torch.rand((3, self.patch_size, self.patch_size))for m in supervise_matches]

            for j, ex in enumerate(supervise_matches):
                vutils.save_image(ex, smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/iter_{iter}/layer_{self.name}/supervise_matches') + f'centroid_{j}.png', normalize=True)
                plt.bar(np.arange(len(self.cent_g[j, :])), self.cent_g[j, :])
                plt.xlabel('Classes')
                plt.ylabel('CIN %')
                plt.title(f'Centroid {j} CIN')
                plt.savefig(smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/iter_{iter}/layer_{self.name}/supervise_cin') + f'centroid_{j}.png')
                plt.close()


    def classify(self, X, Y, task, iter):
        self.model.eval()
        self.model.backbone.eval()

        # for each image
        v_l = torch.zeros((len(X), len(np.unique(Y))))
        labels = []
        
        cin_inds = np.argwhere(np.amax(self.cent_g, axis=1)).flatten()
        
        for it, x in enumerate(X):
            z, p = self.extract_patch_embeddings(x, return_raws=True)

            labels.append(Y[it])
            
            d = 1 - pairwise_cosine_similarity(z, self.ltm).detach().cpu()
            #d = 1 - pairwise_cosine_similarity(z, self.ltm[cin_inds]).detach().cpu()

            close_ind = torch.argmin(d, dim = 1)

            votes = np.amax(self.cent_g[close_ind, :])
            if votes < self.rho_task:
                votes = 0.0

            vote_class = np.argmax(self.cent_g[close_ind, :])

            v_l[it, vote_class] += votes
            
            #print(len(self.ltm_examples))
            #print(close_ind)
            #print([len(self.ltm_examples[id]) for id in close_ind])
            #cent_p = torch.stack([self.ltm_examples[id][0] if (id < len(self.ltm_examples)) else torch.rand((3, self.patch_size, self.patch_size)) for id in close_ind])
            #if self.plot:
            #    self.save_test_patches(p, cent_p, votes, vote_class, v_l[it], Y[it], it, task, iter)

        return v_l, labels
    
    def embed(self, x):
        if len(x.shape) > 3:
            x = x.squeeze()

        z = self.extract_patch_embeddings(x)
        d = 1 - pairwise_cosine_similarity(z, self.ltm).detach().cpu()

        close_ind = torch.argmin(d, dim = 1)

        return close_ind.numpy().flatten()

    def init_memory(self, trainloader):
        print('Initializaing Memory')
        self.stm_matches = torch.zeros(self.delta).to('cuda')
        self.stm_unused_count = torch.zeros(self.delta).to('cuda')

        self.stm_ages = torch.zeros(self.delta).to('cuda')
        self.stm_examples = []


        self.ltm = torch.zeros((0, 512)).to('cuda')
        self.ltm_task = torch.zeros(0).to('cuda')
        self.ltm_examples = []

        embeddings = []
        raws = []
        it = 0
        for batch, t, _ in tqdm(trainloader):
            if it > len(trainloader) // self.init_epochs:
                break
            x = t.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)#.squeeze()
            x = x.permute(0, 2, 3, 1, 4, 5)
            x = x.reshape(-1, 3, self.patch_size, self.patch_size)
            inds = np.random.choice(len(x), int(len(x) * self.init_sample_factor), replace=False)
            z = self.model.embed(x[inds].to('cuda')).detach()
            raws.append(x[inds])
            embeddings.append(z)
            it += 1

        embeddings = torch.cat(embeddings)
        raws = torch.cat(raws)

        cl, c = KMeans_cosine(embeddings.to('cuda'), self.init_clusters, 1500)

        #clusterer = AffinityPropagation()
        #cl = clusterer.fit_predict(embeddings)
        #c = clusterer.cluster_centers_

        K = len(torch.unique(cl))
        print('K: ', K)

        for i, clust in enumerate(torch.unique(cl)):
            # Get the inputs that matched with the cluster
            clust_inds = torch.argwhere(cl == clust).flatten().to('cuda')

            self.ltm_examples.append([raws[ind] for ind in clust_inds[:self.M // 2]])
            vutils.save_image(raws[clust_inds.detach().cpu()], smart_dir(f'logs/{self.log}/{self.dataset}/task_0/iter_0/layer_{self.name}/ltm_clusters') + f'cluster_{clust}_imgs.png', normalize=True)


        # Estimate the distance threshold
        inds = torch.Tensor(np.random.choice(len(embeddings), 2000, replace=False)).to(torch.int64).to('cuda')
        #dists_temp = 1 - pairwise_cosine_similarity(embeddings.to('cuda')[inds], c[cl[inds]])
        #dists = torch.diagonal(dists_temp)
        dists = 1 - pairwise_cosine_similarity(embeddings.to('cuda')[inds]).cpu()
        dists += 2 * torch.eye(len(dists))
        dists = torch.amin(dists, dim=1)
        self.window.extend(dists.detach().cpu().numpy().tolist())
        self.distance_threshold = perc(np.sort(np.array(self.window)), self.beta)

        # Create the LTM
        self.ltm = c.detach().clone().to('cuda')

        # Create the STM
        inds = torch.tensor(np.random.choice(torch.unique(cl).detach().cpu().numpy(), self.delta, replace=True)).to('cuda')
        self.stm = self.ltm.clone()[inds]
        for clust in inds:
            clust_inds = torch.argwhere(cl == clust).flatten().to('cuda')
            self.stm_examples.append([raws[clust_inds[0]]])

        # Fix any zero centroids (didn't get used)
        cent_sums = torch.sum(self.ltm, dim=1)
        inds = torch.argwhere(cent_sums < .99).flatten()
        for ind in inds:
            self.ltm[ind] = torch.ones(self.ltm.shape[1]).to('cuda') / self.ltm.shape[0]

        # Visualization
        self.distance_threshold_history.append(self.distance_threshold)
        self.step = 0

    def update_memory(self, mem_ex, task):
        if self.mem_update == 'reset':
            self.update_memory_1(mem_ex, task)
        elif self.mem_update == 'reduce_mem':
            self.update_memory_3(mem_ex, task)
        else:
            self.update_memory_2(mem_ex, task)
        
    
    def update_memory_1(self, mem_exs, task):
        stm_sizes = np.array([len(ex) for ex in self.stm_examples])
        self.stm = torch.stack([torch.mean(self.model.embed(torch.stack(self.stm_examples[i])[:self.update_use].to('cuda')), dim=0) for i in np.argwhere(stm_sizes >= self.update_use).flatten()])
        self.ltm = torch.stack([torch.mean(self.model.embed(torch.stack(ex)[:self.update_use].to('cuda')), dim=0) for ex in self.ltm_examples])

        self.stm_examples = [[self.stm_examples[i][0]] for i in range(len(self.stm_examples))]
        self.stm_matches[:] = 0

        self.window = deque(maxlen=self.window_size)
        embeddings = []
        for it, ex in enumerate(mem_exs):
            embeddings.append(self.model.embed(ex[None, :].to('cuda')))

        print('Resetting Novelty')
        embeddings = torch.cat(embeddings)
        inds = torch.Tensor(np.random.choice(len(embeddings), 2000, replace=False)).to(torch.int64).to('cuda')
        dists = 1 - pairwise_cosine_similarity(embeddings[inds]).cpu()
        dists += 2 * torch.eye(len(dists))
        dists = torch.amin(dists, dim=1)
        self.window.extend(dists.detach().cpu().numpy().tolist())
        self.distance_threshold = perc(np.sort(np.array(self.window)), self.beta)

    def update_memory_2(self, mem_exs, task):
        stm_sizes = np.array([len(ex) for ex in self.stm_examples])

        self.stm = torch.stack([torch.mean(self.model.embed(torch.stack(self.stm_examples[i])[:self.update_use].to('cuda')), dim=0) for i in np.argwhere(stm_sizes >= self.update_use).flatten()])
        self.ltm = torch.stack([torch.mean(self.model.embed(torch.stack(ex)[:self.update_use].to('cuda')), dim=0) for ex in self.ltm_examples])

        self.stm_examples = [[self.stm_examples[i][0]] for i in range(len(self.stm_examples))]
        self.stm_matches[:] = 0

    
    def update_memory_3(self, mem_ex, task):
        print('Reduce Memory Update')
        stm_sizes = np.array([len(ex) for ex in self.stm_examples])

        self.stm = torch.stack([torch.mean(self.model.embed(torch.stack(self.stm_examples[i])[:self.update_use].to('cuda')), dim=0) for i in np.argwhere(stm_sizes >= self.update_use).flatten()])
        self.ltm = torch.stack([torch.mean(self.model.embed(torch.stack(ex)[:self.update_use].to('cuda')), dim=0) for ex in self.ltm_examples])

        self.stm_examples = [[self.stm_examples[i][0]] for i in range(len(self.stm_examples))]
        self.stm_matches[:] = 0

        # Rework LTM Examples
        for c, cent in enumerate(self.ltm):
            if (len(self.ltm_examples[c]) / self.M)**self.forgetting_factor < np.random.uniform(0, 1):
                z = self.model.embed(torch.stack(self.ltm_examples[c]).to('cuda'))
                dists = 1 - pairwise_cosine_similarity(z, z)
                dists += 2 * torch.eye(len(dists))
                ind = torch.argmin(dists)
                ind_row = ind // dists.shape[0]
                ind_col = ind % dists.shape[1]
                del self.ltm_examples[c][max(ind_row, ind_col)]


    def pretrain(self, dataloader):
        self.model.train()
        self.model.backbone.train()
        dataset = dataloader.dataset
        sampler = dataloader.sampler

        collate_fn = self.pre_collate_fn
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.pretrain_bs,
            collate_fn=collate_fn,
            drop_last=True,
            sampler=sampler,
            num_workers=self.n_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        log = f'logs/{self.pretrain_log}/{self.dataset}/task_0/layer_{self.name}/saved_models/final_model.pkl'
        print(os.path.exists(log), log)
        if self.load_pretrain and os.path.exists(log):
            print('Loading Pretrain')
            self.model.load_state_dict(torch.load(log))

        else:
            optimizer, scheduler = self.model.configure_optimizers(self.lr, self.wd, self.init_epochs, len(trainloader) * self.init_epochs)
            self.model.train()
            print(f"Starting Pretraining - Task 0 - Layer: {self.name}")
            ite = 0
            loss_history = []
            #for epoch in range(epochs):
            total_loss = 0
            for batch, t, _ in tqdm(trainloader):

                if ite % len(trainloader) == 0:
                    vutils.save_image(batch[0][:], smart_dir(f'logs/{self.log}/{self.dataset}/task_0/layer_{self.name}') + 'aug_0.png', normalize=True)
                    vutils.save_image(batch[1][:], smart_dir(f'logs/{self.log}/{self.dataset}/task_0/layer_{self.name}') + 'aug_1.png', normalize=True)

                loss = self.model.training_step(batch)

                loss.backward()

                total_loss += loss.item()

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
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
                
            torch.save(self.model.state_dict(), smart_dir(f'logs/{self.log}/{self.dataset}/task_0/layer_{self.name}/saved_models') + f'final_model.pkl')

        self.model.eval()
        self.model.backbone.eval()
        self.init_memory(trainloader)


    def sleep(self, task):

        if not self.sleep_on:
            print('NOT SLEEPING')
            return 
        
        self.model.train()
        self.model.backbone.train()
        
        stm_ex_sizes = [len(self.stm_examples[i]) for i in range(len(self.stm_examples))]
        ltm_ex_sizes = [len(self.ltm_examples[i]) for i in range(len(self.ltm_examples))]
        self.ltm_mem_size.append(np.sum(ltm_ex_sizes))
        self.stm_mem_size.append(np.sum(stm_ex_sizes))
        
        plt.plot(self.stm_mem_size, 'o-', label='STM MEM Size')
        plt.plot(self.ltm_mem_size, 'o-', label='LTM MEM Size')
        plt.title('Memory Size over Time')
        plt.xlabel('Sleep Cycles')
        plt.ylabel('Total Examples')
        plt.legend()
        plt.grid()
        plt.savefig(smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/sleep_{self.sleep_cycles}/layer_{self.name}/figures') + 'memory_size_over_time.png')
        plt.close()
        
        plt.hist(stm_ex_sizes)
        plt.savefig(smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/sleep_{self.sleep_cycles}/layer_{self.name}/figures') + 'stm_sizes_hist.png')
        plt.close()

        for i, stm_example in enumerate(self.stm_examples):
            if len(stm_example) > 0:
                vutils.save_image(stm_example, smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/sleep_{self.sleep_cycles}/layer_{self.name}/stm_clusters') + f'cluster_{i}_imgs.png', normalize=True)


        loss_history = []

        stm_sizes = np.array([len(ex) for ex in self.stm_examples])
        stm_ex = torch.cat([torch.stack(self.stm_examples[i])[:self.theta] for i in np.argwhere(stm_sizes > 1).flatten()])
        ltm_ex = torch.cat([torch.stack(ex) for ex in self.ltm_examples])
        labs = np.concatenate((np.repeat(self.sleep_cycles, len(stm_ex)), np.repeat(-1, len(ltm_ex))))
        mem_ex = torch.cat((stm_ex, ltm_ex))
        dataset = NumpyDataset(mem_ex, np.repeat(-1, len(mem_ex)))

        #epochs = self.sleep_epochs

        dataset = LightlyDataset.from_torch_dataset(dataset)

        collate_fn = self.sleep_collate_fn
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.sleep_bs,
            collate_fn=collate_fn,
            drop_last=True,
            sampler=ExtendedSampler(np.arange(len(dataset.dataset.labels)), shuffle=True, repeats=self.sleep_epochs),
            num_workers=self.n_workers,
            persistent_workers=True,
            pin_memory=True,
        )
            
        print('TRAIN LENGTH: ', len(trainloader))
        if self.load_sleep:
            print('Loading Sleep')
            log = f'logs/{self.sleep_log}/{self.dataset}/task_{task}/sleep_{self.sleep_cycles}/layer_{self.name}/saved_models/final_model.pkl'
            self.model.load_state_dict(torch.load(log))
        else:
            optimizer, scheduler = self.model.configure_optimizers(self.lr, self.wd, self.sleep_epochs, len(trainloader) * self.sleep_epochs)
            self.model.train()
            print(f"Starting Sleep - Task {task} - Cycle {self.sleep_cycles} - Layer: {self.name}")
            ite = 0
            #for epoch in range(epochs):
            total_loss = 0
            for batch, t, _ in tqdm(trainloader):

                if ite % len(trainloader) == 0:
                    vutils.save_image(batch[0][:], smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/sleep_{self.sleep_cycles}/layer_{self.name}') + 'aug_0.png', normalize=True)
                    vutils.save_image(batch[1][:], smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/sleep_{self.sleep_cycles}/layer_{self.name}') + 'aug_1.png', normalize=True)

                loss = self.model.training_step(batch)

                loss.backward()

                total_loss += loss.item()

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                ite += 1

                if ite % (len(trainloader) // self.sleep_epochs) == 0:
                    avg_loss = total_loss / (len(trainloader) // self.sleep_epochs)
                    total_loss = 0
                    print(f'Epoch: [{ite}/{len(trainloader)}],   Loss: {avg_loss}')
                    loss_history.append(avg_loss)

                    plt.plot(loss_history)
                    plt.title('Average Loss over Training')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.savefig(smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/sleep_{self.sleep_cycles}/layer_{self.name}/figures') + 'loss_history.png')
                    plt.close()

            torch.save(self.model.state_dict(), smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/sleep_{self.sleep_cycles}/layer_{self.name}/saved_models') + f'final_model.pkl')

        self.model.eval()
        self.model.backbone.eval()

        self.update_memory(mem_ex, task)
        
        self.sleep_cycles += 1

        return 

    def __call__(self, x, t):
        #x = x.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride).reshape(1, 3, -1, self.patch_size, self.patch_size).swapaxes(1, 2).squeeze()

        z = self.model.embed(x.to('cuda'))

        distances = 1 - pairwise_cosine_similarity(z, torch.cat((self.stm, self.ltm)))

        # Index of closest centroid for each patch
        close_ind_z = torch.argmin(distances, dim=1)
        
        # Distance of closest centroid for each patch
        close_val_z = torch.amin(distances, dim=1)

        # Novelty
        # Indices of patches who are novel and are in the STM
        novel_inds = torch.argwhere((close_val_z > self.distance_threshold) & (close_ind_z <= len(self.stm))).flatten().tolist()
        num_novelties = len(novel_inds)

        self.nd_history.append(num_novelties)

        evict_inds = []
        if num_novelties > 0:

            # Indices of stm most ready to be evicted
            evict_inds = torch.argsort(self.stm_ages)[-num_novelties:]

            self.stm[evict_inds] = z[novel_inds]
            self.stm_matches[evict_inds] = 1
            self.stm_ages[evict_inds] = 0
            for i in range(len(evict_inds)):
                self.stm_examples[evict_inds[i]] = [x[novel_inds[i]]]

            
        
        # The indices of patches that matched with something
        z_match_inds = set(torch.argwhere(close_val_z <= self.distance_threshold).flatten().tolist())

        # Filter out the LTM matches
        z_match_inds = list(
            z_match_inds.intersection(
                set(
                    torch.argwhere(close_ind_z < self.delta).flatten().tolist()
                )
            ).intersection(
                set(
                    torch.argwhere(self.stm_matches >= 0).flatten().tolist()
                )
            )
        )

        # The indices of the STM for these filtered matches
        stm_match_inds = close_ind_z[z_match_inds]
            
        
        if len(z_match_inds) > 0:
            # Update the centroid position
            self.stm[stm_match_inds] = (1-self.alpha) * self.stm[stm_match_inds] \
                + self.alpha * z[z_match_inds]
            
            # Add the matched examples to the STM example memory
            for i in range(len(stm_match_inds)):
                if len(self.stm_examples[stm_match_inds[i]]) < self.theta:
                    self.stm_examples[stm_match_inds[i]].append(x[z_match_inds[i]])
            
            self.stm_matches[stm_match_inds] += 1


        self.stm_ages += 1

        #######################
        ## Novelty Detection ##
        #######################


        # Sample num_distance_samples distances to add to sliding window
        sampled_d = np.random.choice(close_val_z.cpu().numpy(), self.num_distance_samples)
        # Add samples to sliding window
        self.window.extend(sampled_d)
        # Update Distance Threshold
        self.distance_threshold = perc(np.sort(np.array(self.window)), self.beta)

        # Visualization
        self.distance_threshold_history.append(self.distance_threshold)
        self.step += 1

        ###########
        ### LTM ###
        ###########


        # Indices of mature STM centroids (stm_matches >= self.theta)
        mature_cent_inds = torch.argwhere(self.stm_matches >= self.theta).flatten().detach().cpu().numpy()

        # If there is at least one mature centroid
        if len(mature_cent_inds) > 0:
            # Set Existing STM Centroid to -1 to mark as not useable (can still be matched with but not updated)    
            self.stm_matches[mature_cent_inds] = -1
            
            # Create a new  LTM centroid
            self.ltm = torch.cat((self.ltm, torch.clone(self.stm[mature_cent_inds])), dim = 0)

            for cent in mature_cent_inds:
                self.ltm_examples.append([self.stm_examples[cent][i] for i in range(self.M)])
                self.stm_examples[cent] = [self.stm_examples[cent][0]]
            

        if (self.step == self.sleep_start) or (self.step - self.sleep_start) % self.sleep_freq == 0:
            self.sleep(t)

        self.ltm_size_history.append(len(self.ltm))

        return x
     
    # return ltm centroids
    def get_ltm_centroids(self):
        return self.ltm

    # return stm centroids
    def get_stm_centroids(self):
        return self.stm

    def save_test_patches(self, patches, cent_patches, votes, vote_class, v_l, y, id, task, iter):
        n = int(math.sqrt(len(patches)))
        grid = vutils.make_grid(patches, nrow=n, normalize=True)
        # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

        grid_2 = vutils.make_grid(cent_patches, nrow=n, normalize=True)
        ndarr_2 = grid_2.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,9), gridspec_kw={'width_ratios': [1, 1, 2, 2]})

        # Plot the image
        ax1.imshow(ndarr)
        ax1.axis('off')

        ax2.imshow(ndarr_2)
        ax2.axis('off')
        
        ax3.bar(np.arange(len(votes)), votes)
        ax3.set_xticks(np.arange(len(vote_class)), vote_class)
        ax3.set_xticklabels([class_names[vote_class[i]] for i in range(len(vote_class))], rotation=45)

        ax4.bar(np.arange(len(v_l)), v_l)

        y_pred = np.argmax(v_l)
        plt.title(f'Image {id} Patch Votes\nTrue Label: {class_names[int(y)]}, Pred Label: {class_names[int(y_pred)]}')
        if y == y_pred:
            plt.savefig(smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/iter_{iter}/layer_{self.name}/test_patches/correct') + f'image_{id}.png')
        else:
            plt.savefig(smart_dir(f'logs/{self.log}/{self.dataset}/task_{task}/iter_{iter}/layer_{self.name}/test_patches/incorrect') + f'image_{id}.png')

        plt.close()