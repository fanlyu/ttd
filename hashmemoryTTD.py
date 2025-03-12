import os
from tqdm import tqdm, trange

import torch
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler
import torch.nn as nn

from util.util import info
from util.eval_util import AverageMeter
from models import vision_transformer as vits
from models.l2p_utils.vision_transformer import vit_base_patch16_224_dino

import numpy as np


import os
import pickle 
from collections import defaultdict, Counter
import itertools
import random
from toolkit import get_neighbors, SimpleNN, ContrastiveLoss, real_time_eval, post_eval,DistillationLoss_Centroid

import time


class Hash_Representation:
    def __init__(self, angle_num):
        self.topk = 3
        self.angle_num = angle_num
        self.hash_vectors = torch.randn(angle_num, 768).cpu()
        self.hash_vectors = self.hash_vectors / self.hash_vectors.norm(dim=1, keepdim=True)


        self.hash_centroids = {}    
        self.sim_graph = defaultdict(dict)  
        self.hash_labels = defaultdict(list) 
        self.k = 3

        self.update_hash_graph = HashMemory.update_hash_graph
        self.find_similar_hashes = HashMemory.find_similar_hashes


    def map_to_hash(self, hash_value):
        hash = ''
        
        if hash_value[0] > 5.0:
            hash = hash + '50'
        elif hash_value[0] > 4.8 and hash_value[0] <= 5.0:
            hash = hash + '48'
        elif hash_value[0] > 4.6 and hash_value[0] <= 4.8:
            hash = hash + '46'
        elif hash_value[0] > 4.4 and hash_value[0] <= 4.6:
            hash = hash + '44'
        elif hash_value[0] > 4.2 and hash_value[0] <= 4.4:
            hash = hash + '42'
        elif hash_value[0] > 4.0 and hash_value[0] <= 4.2:
            hash = hash + '40'
        elif hash_value[0] > 3.8 and hash_value[0] <= 4.0:
            hash = hash + '38'
        elif hash_value[0] > 3.6 and hash_value[0] <= 3.8:
            hash = hash + '36'
        elif hash_value[0] > 3.4 and hash_value[0] <= 3.6:
            hash = hash + '34'
        elif hash_value[0] > 3.2 and hash_value[0] <= 3.4:
            hash = hash + '32'
        elif hash_value[0] > 3.0 and hash_value[0] <= 3.2:
            hash = hash + '30'
        elif hash_value[0] > 2.8 and hash_value[0] <= 3.0:
            hash = hash + '28'
        elif hash_value[0] > 2.6 and hash_value[0] <= 2.8:
            hash = hash + '26'
        elif hash_value[0] > 2.4 and hash_value[0] <= 2.6:
            hash = hash + '24'
        elif hash_value[0] > 2.2 and hash_value[0] <= 2.4:
            hash = hash + '22'
        elif hash_value[0] > 0 and hash_value[0] <= 2.2:
            hash = hash + '20'
        else:
            raise Exception("Error Hash Value")
        
        for a in hash_value[1:]:
            hash = hash + str(int(a))
        
        return hash
    
    def merge_dicts_with_samples_and_hashes(self, memory, memory_hash):
        new_dict = {}

        for label in memory:
            samples = memory[label]
            hashes = memory_hash[label]

            if len(samples) != len(hashes):
                raise ValueError(f"label {label} hash and sample are not equal")

            for sample, hash_value in zip(samples, hashes):
                if hash_value not in new_dict:
                    new_dict[hash_value] = []

                new_dict[hash_value].append((sample, label))
        
        return new_dict


class HashMemory:

    def initialize_memory(self):
        self = self
        
        self.knownclass = 70          #70 or 140
        self.totalclass = 100

        self.addclass = 0  


        self.old_classes = list(range(self.knownclass))  
        self.new_classes = list(range(self.knownclass,self.totalclass+self.addclass))
        self.all_classes = list(range(self.totalclass+self.addclass))
        self.data = None

        self.memory = {k:[] for k in self.all_classes}
        self.count = {k:0 for k in self.all_classes}
        self.memory_hash = {k:[] for k in self.all_classes}
        self.mem_per_cls=20


        self.discovered_class = self.knownclass - 1
        self.pred_count = {k: self.totalclass for k in self.old_classes}

        self.centroids = {k: torch.zeros(768, dtype=torch.float32) for k in range(self.totalclass+self.addclass)}

        self.samples_ = {}
        self.images_ = {}


        self.hash_centroids = {}  
        self.sim_graph = defaultdict(dict)  
        self.hash_labels = defaultdict(list) 
        self.k = 3

        

        centroids_filepath='centroids_l2p_cifar70.pkl'
        classes_filepath = 'classes_l2p_cifar70.pkl'
        # centroids_filepath='centroids_l2p_tiny140.pkl'
        # classes_filepath = 'classes_l2p_tiny140.pkl'
        # centroids_filepath = 'centroids_l2p_cub140.pkl'
        # classes_filepath = 'classes_l2p_cub140.pkl'


        with open(centroids_filepath, 'rb') as f:
            loaded_centroids = pickle.load(f)
            for k in range(len(loaded_centroids)):
                self.centroids[k] = torch.tensor(loaded_centroids[k], dtype=torch.float32)

        ''''''
        import numpy as np
        from sklearn.decomposition import PCA

        centroids_matrix = np.stack([self.centroids[k].numpy() for k in range(self.knownclass)], axis=0)


        pca = PCA(n_components=self.hash.angle_num)
        pca.fit(centroids_matrix)  

        principal_components = pca.components_  
        self.hash.hash_vectors = torch.tensor(principal_components, dtype=torch.float32).cpu()
        self.hash.hash_vectors = self.hash.hash_vectors / self.hash.hash_vectors.norm(dim=1, keepdim=True)
        
        ''''''

        for batch in tqdm(self.unlabeled_test_data['default'], desc='Batches', leave=False, bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):
            images, labels, uq_idxs, mask_lab = batch
            with torch.no_grad():
                labels = labels.cpu().numpy()

                self.args.ttd_model == 'TTD_L2P_known_K'
                dino_features = self.original_model(images.cuda())['pre_logits']
                features = self.model(images.cuda(), task_id=self.stage_i, cls_features=dino_features)['x'][:, 0]
                features = features.cpu()

            for x, y, z in zip(features, labels, images):
                y = int(y.item())  
                if y not in self.samples_:
                    self.samples_[y] = []
                    self.images_[y] = []
                self.samples_[y].append(x)
                self.images_[y].append(z)

                if len(self.samples_[y]) > 10000:
                    self.samples_[y].pop(0)
                    self.images_[y].pop(0)


        for label in self.old_classes:
            if label in self.samples_:
                features = self.samples_[label]
                images = self.images_[label]

                shuffled_indices = torch.randperm(len(features))
                selected_indices = shuffled_indices[:self.mem_per_cls]
                
            for idx in selected_indices:
                self.memory[label].append(images[idx].cpu())

                hash_value = [torch.norm(features[idx], p=2, dim=-1)]

                for i in range(self.hash.hash_vectors.shape[0]):
                    hash_value.append(torch.dot(features[idx].squeeze(0), self.hash.hash_vectors[i]) > 0)

                hash_value = self.hash.map_to_hash(hash_value)
                self.memory_hash[label].append(hash_value)

                magnitude_prefix = hash_value[:2]
                angle_bits = hash_value[2:]

                self.hash.update_hash_graph(self, angle_bits, features[idx])



                


            self.count[label] = self.mem_per_cls


    def cos_sim(self, vec1, vec2):
        return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    
    def euclidean_sim(self, vec1,vec2):
        return torch.nn.functional.pairwise_distance(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    


    def update_hash_graph(self, hash_val, feature):    
        cnt = 0
        if hash_val not in self.hash_centroids:
            self.hash_centroids[hash_val] = (feature.clone(), 1)
            for h in self.hash_centroids:
                cnt = cnt + 1
                if h != hash_val:
                    sim = HashMemory.cos_sim(self, feature, self.hash_centroids[h][0])
                    self.sim_graph[hash_val][h] = sim
                    self.sim_graph[h][hash_val] = sim
        else:
            old_centroid, count = self.hash_centroids[hash_val]
            new_centroid = (old_centroid * count + feature) / (count + 1)
            self.hash_centroids[hash_val] = (new_centroid, count + 1)
            
            for h in self.hash_centroids:
                cnt = cnt + 1
                if h != hash_val:
                    sim = HashMemory.cos_sim(self, new_centroid, self.hash_centroids[h][0])
                    self.sim_graph[hash_val][h] = sim
                    self.sim_graph[h][hash_val] = sim

    def find_similar_hashes(self, query_hash):
        if query_hash in self.sim_graph:
            direct_neighbors = list(self.sim_graph[query_hash].keys())
            sorted_neighbors = sorted(direct_neighbors, 
                                     key=lambda x: self.sim_graph[query_hash][x], 
                                     reverse=True)[:self.k]
            return sorted_neighbors
        else:
            return []
        


class TTD_simple:
    def __init__(self, args, model, ori_model, projection_head, data, hash):
        super().__init__()
        self.args = args
        self.discovered_class = 70
        self.model = model
        self.original_model = ori_model
        self.projection_head = projection_head
        self.data = data
        self.hash = hash

        self.hash_centroids = hash.hash_centroids
        self.sim_graph = hash.sim_graph
        self.hash_labels = hash.hash_labels
        self.k = hash.k

        self.update_hash_graph = hash.update_hash_graph
        self.find_similar_hashes = hash.find_similar_hashes

        self.aux_count = 0
        self.record = defaultdict(float)
        self.cen_set = defaultdict(list)
        self.t = 1
        self.cos_count = 0
        self.lsh_count = 0

        self.cluster_labels = {i: [] for i in range(self.discovered_class)}

        self.optimizer = SGD(
            list(self.projection_head.parameters()) + list(self.model.parameters()), 
            lr=self.args.base_lr, 
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        self.exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=self.args.base_lr * 1e-3,
        )

    def extract_features_and_labels(self, unlabeled_test_data, stage_i):
        self.stage_i = stage_i
        all_feats = []
        all_labels = []


        for batch in tqdm(unlabeled_test_data['default'], desc='Batches', leave=False, bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):
            images, labels, uq_idxs, mask_lab = batch

            with torch.no_grad():
                self.args.ttd_model == 'TTD_L2P_known_K'
                dino_features = self.original_model(images.cuda())['pre_logits']
                feat = self.model(images.cuda(), task_id=self.stage_i, cls_features=dino_features)['x'][:, 0]
 
            all_feats.append(feat.cpu().numpy())
            all_labels.append(labels.cpu().numpy()) 

        all_feats = np.concatenate(all_feats, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

    def predict_and_discover(self,unlabeled_test_data, stage_i, TTT=False, replay=True, self_correction=True):
        self.stage_i = stage_i
        self.model.eval()
        input_num = 0
        a_known_acc, a_truelabel_agreement_ratio, a_truelabel_entropy, a_cluster_agreement_ratio, a_cluster_entropy = 0,0,0,0,0

        k = 1
        from collections import deque
        pred_eval = deque(maxlen=10*self.args.batch_size)
        label_eval = deque(maxlen=10)


        # post_eval(self, self.args, self.data, unlabeled_test_data)
        print("############ pre_eval finished ##############")
        batchnum = 1
        for batch in tqdm(unlabeled_test_data['default'], desc='Batches', leave=False, bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):
            T3 = time.perf_counter()
            ############ 1. forward and obtain features ###################
            inputs, labels, uq_idxs, mask_lab = batch 

            # images, labels, *other = self.unlabeled_test_data['default']         
            
            with torch.no_grad():

                self.args.ttd_model == 'TTD_L2P_known_K'
                dino_features = self.original_model(inputs.cuda())['pre_logits']
                feats = self.model(inputs.cuda(), task_id=self.stage_i, cls_features=dino_features)['x'][:, 0]
            
            feat_cpu = feats.cpu().detach().numpy()
            feat_norms = np.linalg.norm(feat_cpu, axis=1, keepdims=True)

            ############ 2. predict and discover novel class ##############
            # preds = self.predict_and_discover_with_Euclidean_distance(inputs, feats, replay=False, threshold=5)
            # preds = self.predict_and_discover_with_magitude(feats)

            # preds = self.predict_and_discover_with_cosine_similarity(inputs, feats, replay=False)
            # preds = self.predict_and_discover_with_entropy(inputs, feats, replay)

            batchnum = batchnum + 1
            preds = self.predict_and_discover_with_cosine_and_lsh(inputs, feats, replay=replay)


            for i in range(len(preds)):
                if preds[i] not in self.cluster_labels:
                    self.cluster_labels[preds[i]] = []
                self.cluster_labels[preds[i]].append(labels[i].item())

            pred_eval.extend(preds)
            label_eval.append(labels)

            if k > 9:
                all_preds = list(pred_eval)  
                all_labels = torch.cat(list(label_eval))  
                real_time_eval(self.data, self.record, pred_eval, all_labels)
                
            k = k + 1

            ############ 3. test time training ############################
            if TTT:
                self.test_time_training(replay, feats, preds)
                    
        
            if self_correction:
                if k > 9 and (k % 2 == 0):
                    self.self_memory_correction()

            del inputs, labels, uq_idxs, mask_lab, feats
            torch.cuda.empty_cache()
            T4 = time.perf_counter()

        ############ 5. post evaluation ####################
        post_eval(self, self.args, self.data, unlabeled_test_data)

        self.model.eval()
        all_feats = []
        all_labels = []

        for inputs, labels, uq_idxs, mask_lab in tqdm(unlabeled_test_data['default']):
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                self.args.ttd_model == 'TTD_L2P_known_K'
                dino_features = self.original_model(inputs.cuda())['pre_logits']
                feats = self.model(inputs.cuda(), task_id=self.stage_i, cls_features=dino_features)['x'][:, 0]

            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.cpu().numpy()) 

        all_feats = np.concatenate(all_feats, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)


    def test_time_training(self, replay, feats, preds):
        self.projection_head.train()
        self.model.train(True)

        for label in self.data.memory.keys():
            self.data.memory[label] = [tensor.cpu() for tensor in self.data.memory[label]]

        feats_ori = feats
        preds_ori = preds

        max_samples_per_label = 2
        max_total_replay_samples = 20 

        if replay:
            replay_samples, replay_labels = [], []
            for label, samples in self.data.memory.items():
                limited_samples = random.sample(samples, min(max_samples_per_label, len(samples))) 
                replay_samples.extend(limited_samples)
                replay_labels.extend([label] * len(limited_samples))

            if len(replay_samples) > max_total_replay_samples:
                combined = list(zip(replay_samples, replay_labels))
                random.shuffle(combined)
                replay_samples, replay_labels = zip(*combined[:max_total_replay_samples])
                replay_samples = list(replay_samples)
                replay_labels = list(replay_labels)

            replay_samples_tensor = [replay_tensor.to('cuda') for replay_tensor in replay_samples]                    
            replay_samples_tensor = torch.stack(replay_samples)
            replay_labels = list(replay_labels)

            self.args.ttd_model == 'TTD_L2P_known_K'
            dino_features = self.original_model(replay_samples_tensor.cuda())['pre_logits']
            replay_feats = self.model(replay_samples_tensor.cuda(), task_id=self.stage_i, cls_features=dino_features)['x'][:, 0]

            feats = torch.concat([feats, replay_feats], dim= 0)
            preds = preds + replay_labels

        loss = self.data.get_criterion_test(feats, preds, self.data.centroids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def self_memory_correction(self):
        replay_samples, replay_labels, original_hash = [], [], []

        for label in range(self.data.knownclass,self.data.discovered_class + 1):
            idx = random.randint(0, len(self.data.memory_hash[label])-1)
            replay_samples.append(self.data.memory[label][idx])
            original_hash.append(self.data.memory_hash[label][idx])
            replay_labels.extend([label])

            if(len(self.data.memory[label]) > 1):
                del self.data.memory[label][idx]
                del self.data.memory_hash[label][idx]
        
        replay_samples = list(replay_samples)
        replay_labels = list(replay_labels)
        original_hash = list(original_hash)

        replay_samples_tensor = [replay_tensor.to('cuda') for replay_tensor in replay_samples]                    
        replay_samples_tensor = torch.stack(replay_samples)
        replay_labels = list(replay_labels)


        with torch.no_grad():
            self.args.ttd_model == 'TTD_L2P_known_K'
            dino_features = self.original_model(replay_samples_tensor.cuda())['pre_logits']
            updated_features = self.model(replay_samples_tensor.cuda(), task_id=self.stage_i, cls_features=dino_features)['x'][:, 0]

        for x, feat in zip(replay_samples, updated_features):
            feat = feat.cpu()
            feat_norm = torch.norm(feat, p=2)
            hash_value = [torch.norm(feat, p=2)]
            for i in range(self.hash.hash_vectors.shape[0]):
                hash_value.append(torch.dot(feat.squeeze(0).cpu(), self.hash.hash_vectors[i])>0)
            hash_value = self.hash.map_to_hash(hash_value)
        
            hash_dict = self.hash.merge_dicts_with_samples_and_hashes(self.data.memory, self.data.memory_hash)
            inputs, labels = [], []

            magnitude_prefix = hash_value[:2]
            angle_bits = hash_value[2:]

            neighbor_buckets_old = []
            neighbor_buckets = []
            angle_buckets = self.find_similar_hashes(self, angle_bits) + [angle_bits]

            for angle_neighbor in angle_buckets:
                neighbor_buckets_old.append(magnitude_prefix + angle_neighbor)
            
            for hash_neighbor in neighbor_buckets_old:
                neighbor_buckets = neighbor_buckets + get_neighbors(hash_neighbor)
            if len(angle_buckets)>=1:
                i = 1
                for _hash_value in neighbor_buckets:
                    i = i + 1
                    if _hash_value in hash_dict.keys():
                        bucket_inputs, bucket_labels = zip(*hash_dict[_hash_value])
                        inputs.extend(bucket_inputs) 
                        labels.extend(bucket_labels) 

            if len(inputs) > 0:
                if(len(inputs) > 40):
                    combined = list(zip(inputs, labels))
                    random.shuffle(combined)
                    inputs, labels = zip(*combined[:40]) 
                    inputs = list(inputs)
                    labels = list(labels)

                inputs = [input_tensor.to('cuda') for input_tensor in inputs]

                inputs = torch.stack(inputs)
                labels = torch.tensor(labels).to('cuda') 
            
                with torch.no_grad():

                    self.args.ttd_model == 'TTD_L2P_known_K'
                    dino_features = self.original_model(inputs)['pre_logits']
                    feat_queries = self.model(inputs, task_id=self.stage_i, cls_features=dino_features)['x'][:, 0]
                
                feat_queries = feat_queries.cpu()
                distances = torch.norm(feat_queries - feat.unsqueeze(0), p=2, dim=1)
                distances_labels = list(zip(distances.tolist(), labels.tolist()))

                k = min(10, len(inputs))
                nearest = sorted(distances_labels, key=lambda x: x[0])[:k]

                nearest_labels = [label for _, label in nearest]
                pred = Counter(nearest_labels).most_common(1)[0][0]

                if nearest[0][0] <= 3:
                    if pred in self.data.new_classes:
                        if len(self.data.memory[pred]) < self.data.mem_per_cls:
                            self.data.memory[pred].append(x)
                            self.data.memory_hash[pred].append(hash_value)
                        else:
                            idx = random.randint(0, self.data.count[pred]-1)
                            if idx < self.data.mem_per_cls: 
                                self.data.memory[pred][idx] = x
                                self.data.memory_hash[pred][idx] = hash_value

    
    def predict_and_discover_with_Euclidean_distance(self, inputs, feats, replay=False, only_test=False, threshold=2.8):
        preds = []
        centroid_cache = {}
        centroid_cache_num = {}
        for x, feat in zip(inputs, feats):
            pred = min(self.data.centroids.keys(),key=lambda k: torch.norm(torch.tensor(self.data.centroids[k]).to('cuda') - feat, p=2))
            min_distance = torch.norm(torch.tensor(self.data.centroids[pred]).to('cuda') - feat, p=2).item()

            if only_test:
                preds.append(pred)
                continue

            if min_distance <= threshold or self.data.discovered_class >= self.data.totalclass - 1: 
                if replay:
                    if len(self.data.memory[pred]) < self.data.mem_per_cls:
                        self.data.memory[pred].append(x)
                    else:
                        idx = random.randint(0, self.data.count[pred]-1)
                        if idx < self.data.mem_per_cls: 
                            self.data.memory[pred][idx] = x
                    self.data.count[pred] += 1
            else:
                self.data.discovered_class += 1
                pred = self.data.discovered_class
                self.data.centroids[pred] = feat
                if replay:
                    self.data.count[pred] = 1
                    self.data.memory[pred] = [x]

            preds.append(pred)
            if pred not in centroid_cache.keys():
                centroid_cache[pred] = feat
                centroid_cache_num[pred] = 1
            else:
                centroid_cache[pred] = centroid_cache[pred] + feat
                centroid_cache_num[pred] = centroid_cache_num[pred] + 1
        
        if not only_test:
            cen_set = []
            for key in centroid_cache.keys(): 
                if key in self.data.old_classes:
                    self.data.centroids[key] = 1 * torch.tensor(self.data.centroids[key]).to('cuda') + 0 * (centroid_cache[key] / centroid_cache_num[key])
                else:
                    self.data.centroids[key] = 0.9 * torch.tensor(self.data.centroids[key]).to('cuda') + 0.1 * (centroid_cache[key] / centroid_cache_num[key])

                cen_set.append(torch.norm(self.data.centroids[key],p=2).detach().cpu().numpy())

        return preds

    def predict_and_discover_with_magitude(self, feats, only_test = False, threshold=3):
        preds = []
        centroid_cache = {}
        centroid_cache_num = {}
        for feat in feats:
            feat_mag = torch.norm(feat, p=2)

            if only_test:
                pred = min(self.data.centroids.keys(),key=lambda k: torch.norm(torch.tensor(self.data.centroids[k]).to('cuda') - feat, p=2))
                preds.append(pred)
                continue

            if feat_mag >= threshold or self.data.discovered_class >= self.data.totalclass - 1:
                pred = min(self.data.centroids.keys(),key=lambda k: torch.norm(torch.tensor(self.data.centroids[k]).to('cuda') - feat, p=2))
            else:
                self.data.discovered_class += 1
                pred = self.data.discovered_class
                self.data.centroids[pred] = feat
            preds.append(pred)
            if pred not in centroid_cache.keys():
                centroid_cache[pred] = feat
                centroid_cache_num[pred] = 1
            else:
                centroid_cache[pred] = centroid_cache[pred] + feat
                centroid_cache_num[pred] = centroid_cache_num[pred] + 1
        
        if not only_test:
            cen_set = []
            for key in centroid_cache.keys(): 
                if key in self.data.old_classes:
                    self.data.centroids[key] = 1 * torch.tensor(self.data.centroids[key]).to('cuda') + 0 * (centroid_cache[key] / centroid_cache_num[key])
                else:
                    self.data.centroids[key] = 0.9 * torch.tensor(self.data.centroids[key]).to('cuda') + 0.1 * (centroid_cache[key] / centroid_cache_num[key])

                cen_set.append(torch.norm(self.data.centroids[key],p=2).detach().cpu().numpy())

        return preds   


    def predict_and_discover_with_cosine_similarity(self, inputs, feats, replay=False, only_test=False, threshold=0.7):
        preds = []
        centroid_cache = {}
        centroid_cache_num = {}
        
        for x, feat in zip(inputs, feats):
            feat_norm = torch.norm(feat, p=2)
            similarities = {k: torch.dot(torch.tensor(self.data.centroids[k]).to('cuda'), feat) / 
                                (torch.norm(torch.tensor(self.data.centroids[k]).to('cuda'), p=2) * feat_norm) 
                            for k in self.data.centroids.keys()}

            pred = max(similarities, key=similarities.get)
            max_similarity = similarities[pred].item()

            if only_test:
                preds.append(pred)
                continue

            if max_similarity >= threshold or self.data.discovered_class >= self.data.totalclass - 1 + self.data.addclass:
                if replay:
                    if len(self.data.memory[pred]) < self.data.mem_per_cls:
                        self.data.memory[pred].append(x)
                    else:
                        idx = random.randint(0, self.data.count[pred] - 1)
                        if idx < self.data.mem_per_cls: 
                            self.data.memory[pred][idx] = x
                    self.data.count[pred] += 1
            else:
                self.data.discovered_class += 1
                pred = self.data.discovered_class
                self.data.centroids[pred] = feat
                if replay:
                    self.data.count[pred] = 1
                    self.data.memory[pred] = [x]

            preds.append(pred)

            if pred not in centroid_cache.keys():
                centroid_cache[pred] = feat
                centroid_cache_num[pred] = 1
            else:
                centroid_cache[pred] = centroid_cache[pred] + feat
                centroid_cache_num[pred] += 1

        if not only_test:
            for key in centroid_cache.keys():
                if key in self.data.old_classes:
                    self.data.centroids[key] = 1 * torch.tensor(self.data.centroids[key]).to('cuda') + 0 * (centroid_cache[key] / centroid_cache_num[key])
                else:
                    self.data.centroids[key] = 0.9 * torch.tensor(self.data.centroids[key]).to('cuda') + 0.1 * (centroid_cache[key] / centroid_cache_num[key])
        return preds

    def predict_and_discover_with_cosine_similarity_plus(self, inputs, feats, replay=False, only_test=False, threshold=0.8, thres_eu=2.8):
        preds = []
        centroid_cache = {}
        centroid_cache_num = {}
        
        for x, feat in zip(inputs, feats):
            feat_norm = torch.norm(feat, p=2)
            similarities = {k: torch.dot(torch.tensor(self.data.centroids[k]).to('cuda'), feat) / 
                                (torch.norm(torch.tensor(self.data.centroids[k]).to('cuda'), p=2) * feat_norm) 
                            for k in self.data.centroids.keys()}

            pred = max(similarities, key=similarities.get)
            max_similarity = similarities[pred].item()

            min_distance = torch.norm(torch.tensor(self.data.centroids[pred]).to('cuda') - feat, p=2).item()

            if only_test:
                preds.append(pred)
                continue

            if (max_similarity >= threshold or min_distance <= thres_eu):
                if replay:
                    if len(self.data.memory[pred]) < self.data.mem_per_cls:
                        self.data.memory[pred].append(x)
                    else:
                        idx = random.randint(0, self.data.count[pred] - 1)
                        if idx < self.data.mem_per_cls: 
                            self.data.memory[pred][idx] = x
                    self.data.count[pred] += 1


                if (max_similarity < 0.7 and min_distance <= thres_eu):
                    pred = min(self.data.centroids.keys(),key=lambda k: torch.norm(torch.tensor(self.data.centroids[k]).to('cuda') - feat, p=2))

            else:
                if(min_distance <= thres_eu):
                    pred = min(self.data.centroids.keys(),key=lambda k: torch.norm(torch.tensor(self.data.centroids[k]).to('cuda') - feat, p=2))
                if(max_similarity >= threshold):
                    pred = max(similarities, key=similarities.get)

                self.data.discovered_class += 1
                pred = self.data.discovered_class
                self.data.centroids[pred] = feat
                if replay:
                    self.data.count[pred] = 1
                    self.data.memory[pred] = [x]

            preds.append(pred)

            if pred not in centroid_cache.keys():
                centroid_cache[pred] = feat
                centroid_cache_num[pred] = 1
            else:
                centroid_cache[pred] = centroid_cache[pred] + feat
                centroid_cache_num[pred] += 1

        if not only_test:
            for key in centroid_cache.keys():
                if key in self.data.old_classes:
                    self.data.centroids[key] = 1 * torch.tensor(self.data.centroids[key]).to('cuda') + 0 * (centroid_cache[key] / centroid_cache_num[key])
                else:
                    self.data.centroids[key] = 0.9 * torch.tensor(self.data.centroids[key]).to('cuda') + 0.1 * (centroid_cache[key] / centroid_cache_num[key])
        return preds


    def predict_and_discover_with_entropy(self, inputs, feats, replay, only_test=False, threshold=3.8):
        preds = []
        centroid_cache = {}
        centroid_cache_num = {}
        
        for x, feat in zip(inputs, feats):
            distances = {k: torch.norm(torch.tensor(self.data.centroids[k]).to('cuda') - feat, p=2).item() 
                        for k in self.data.centroids.keys()}
            
            distances_tensor = torch.tensor(list(distances.values()))
            probabilities = torch.exp(-distances_tensor) / torch.sum(torch.exp(-distances_tensor))
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10)).item() 

            if only_test:
                pred = min(distances, key=distances.get)
                preds.append(pred)
                continue

            if entropy <= threshold or self.data.discovered_class >= self.data.totalclass - 1:
                pred = min(distances, key=distances.get)
                if replay:
                    if len(self.data.memory[pred]) < self.data.mem_per_cls:
                        self.data.memory[pred].append(x)
                    else:
                        idx = random.randint(0, self.data.count[pred] - 1)
                        if idx < self.data.mem_per_cls: 
                            self.data.memory[pred][idx] = x
                    self.data.count[pred] += 1
            else:
                self.data.discovered_class += 1
                pred = self.data.discovered_class
                self.data.centroids[pred] = feat
                if replay:
                    self.data.count[pred] = 1
                    self.data.memory[pred] = [x]

            preds.append(pred)

            if pred not in centroid_cache.keys():
                centroid_cache[pred] = feat
                centroid_cache_num[pred] = 1
            else:
                centroid_cache[pred] = centroid_cache[pred] + feat
                centroid_cache_num[pred] += 1

        if not only_test:
            for key in centroid_cache.keys(): 
                if key in self.data.old_classes:
                    self.data.centroids[key] = 1 * torch.tensor(self.data.centroids[key]).to('cuda') + 0 * (centroid_cache[key] / centroid_cache_num[key])
                else:
                    self.data.centroids[key] = 0.9 * torch.tensor(self.data.centroids[key]).to('cuda') + 0.1 * (centroid_cache[key] / centroid_cache_num[key])
                
        return preds


    def predict_and_discover_with_cosine_and_lsh(self, inputs, feats, replay=False, only_test=False, cos_threshold=0.3, lsh_threshold=3.3):
        preds = []
        centroid_cache = {}
        centroid_cache_num = {}

        count_cos = 0
        count_lsh = 0
        
        for x, feat in zip(inputs, feats):
            feat = feat.cpu()
            feat_norm = torch.norm(feat, p=2)
            similarities = {k: torch.dot(torch.tensor(self.data.centroids[k]).to('cpu'), feat) / 
                                (torch.norm(torch.tensor(self.data.centroids[k]).to('cpu'), p=2) * feat_norm) 
                            for k in self.data.centroids.keys()}
            
            pred = max(similarities, key=similarities.get)
            max_similarity = similarities[pred].item()

            pred_cos = pred

            
            hash_value = [torch.norm(feat, p=2)]

            for i in range(self.hash.hash_vectors.shape[0]):
                hash_value.append(torch.dot(feat.squeeze(0), self.hash.hash_vectors[i])>0)

            hash_value = self.hash.map_to_hash(hash_value)
                           
            if(max_similarity > 0.7):
                count_cos = count_cos + 1
                if replay and pred in self.data.new_classes:
                    if len(self.data.memory[pred]) < self.data.mem_per_cls:
                        self.data.memory[pred].append(x)
                        
                        self.data.memory_hash[pred].append(hash_value)
                    else:
                        idx = random.randint(0, self.data.count[pred] - 1)
                        if idx < self.data.mem_per_cls: 
                            self.data.memory[pred][idx] = x
                            
                            self.data.memory_hash[pred][idx] = hash_value
                    self.data.count[pred] += 1
            
                magnitude_prefix = hash_value[:2]
                angle_bits = hash_value[2:]
                    
                self.update_hash_graph(self, angle_bits, feat)

                preds.append(pred)

                if pred not in centroid_cache.keys():
                    centroid_cache[pred] = feat
                    centroid_cache_num[pred] = 1
                else:
                    centroid_cache[pred] = centroid_cache[pred] + feat
                    centroid_cache_num[pred] += 1
                
                self.cos_count += 1
            
            else:
                count_lsh = count_lsh + 1

                hash_dict = self.hash.merge_dicts_with_samples_and_hashes(self.data.memory, self.data.memory_hash)
                inputs, labels = [], []

                magnitude_prefix = hash_value[:2]
                angle_bits = hash_value[2:]

                neighbor_buckets_old = []
                neighbor_buckets = []
                angle_buckets = self.find_similar_hashes(self, angle_bits) + [angle_bits]

                for angle_neighbor in angle_buckets:
                    neighbor_buckets_old.append(magnitude_prefix + angle_neighbor)
                
                for hash_neighbor in neighbor_buckets_old:
                    neighbor_buckets = neighbor_buckets + get_neighbors(hash_neighbor)

                if len(angle_buckets)>=1:
                    i = 1
                    for _hash_value in neighbor_buckets:
                        i = i + 1
                        if _hash_value in hash_dict.keys():
                            bucket_inputs, bucket_labels = zip(*hash_dict[_hash_value])
                            inputs.extend(bucket_inputs)
                            labels.extend(bucket_labels)
                
                self.update_hash_graph(self, angle_bits, feat)

                
                if len(inputs) > 0:
                    if(len(inputs) > 40):
                        combined = list(zip(inputs, labels))
                        random.shuffle(combined)
                        inputs, labels = zip(*combined[:40])
                        inputs = list(inputs)
                        labels = list(labels)

                    inputs = [input_tensor.to('cuda') for input_tensor in inputs]

                    inputs = torch.stack(inputs) 
                    labels = torch.tensor(labels).to('cuda')
                    
                    with torch.no_grad():
                        self.args.ttd_model == 'TTD_L2P_known_K'
                        dino_features = self.original_model(inputs)['pre_logits']
                        feat_queries = self.model(inputs, task_id=self.stage_i, cls_features=dino_features)['x'][:, 0]
                    
                    feat_queries = feat_queries.cpu()
                    distances = torch.norm(feat_queries - feat.unsqueeze(0), p=2, dim=1)
                    distances_labels = list(zip(distances.tolist(), labels.tolist()))

                    k = min(10, len(inputs))
                    nearest = sorted(distances_labels, key=lambda x: x[0])[:k]
                
                    nearest_labels = [label for _, label in nearest]
                    pred = Counter(nearest_labels).most_common(1)[0][0]

                    if nearest[0][0] <= lsh_threshold:
                        if replay and pred in self.data.new_classes:
                        # if replay:
                            if len(self.data.memory[pred]) < self.data.mem_per_cls:
                                self.data.memory[pred].append(x)
                                self.data.memory_hash[pred].append(hash_value)
                            else:
                                idx = random.randint(0, self.data.count[pred]-1)
                                if idx < self.data.mem_per_cls: 
                                    self.data.memory[pred][idx] = x
                                    self.data.memory_hash[pred][idx] = hash_value
                            self.data.count[pred] += 1
                    else:
                        if self.data.discovered_class < self.data.totalclass - 1 + self.data.addclass:
                            self.data.discovered_class += 1
                            pred = self.data.discovered_class
                            self.data.centroids[pred] = feat
                            if replay:
                                self.data.count[pred] = 1
                                self.data.memory[pred] = [x]
                                self.data.memory_hash[pred] = [hash_value]
                        else:
                            pred = pred_cos
                            if replay and pred in self.data.new_classes:
                                if len(self.data.memory[pred]) < self.data.mem_per_cls:
                                    self.data.memory[pred].append(x)
                                    self.data.memory_hash[pred].append(hash_value)
                                else:
                                    idx = random.randint(0, self.data.count[pred]-1)
                                    if idx < self.data.mem_per_cls: 
                                        self.data.memory[pred][idx] = x
                                        self.data.memory_hash[pred][idx] = hash_value
                                self.data.count[pred] += 1
                            self.aux_count += 1
                else:
                    if self.data.discovered_class < self.data.totalclass - 1  + self.data.addclass:
                        self.data.discovered_class += 1
                        pred = self.data.discovered_class
                        self.data.centroids[pred] = feat
                        if replay:
                            self.data.count[pred] = 1
                            self.data.memory[pred] = [x]
                            self.data.memory_hash[pred] = [hash_value]
                    else:
                        pred = pred_cos
                        if replay and pred in self.data.new_classes:
                            if len(self.data.memory[pred]) < self.data.mem_per_cls:
                                self.data.memory[pred].append(x)
                                self.data.memory_hash[pred].append(hash_value)
                            else:
                                idx = random.randint(0, self.data.count[pred]-1)
                                if idx < self.data.mem_per_cls: 
                                    self.data.memory[pred][idx] = x
                                    self.data.memory_hash[pred][idx] = hash_value
                            self.data.count[pred] += 1
                        self.aux_count += 1
                preds.append(pred)
                if pred not in centroid_cache.keys():
                    centroid_cache[pred] = feat
                    centroid_cache_num[pred] = 1
                else:
                    centroid_cache[pred] = centroid_cache[pred] + feat
                    centroid_cache_num[pred] = centroid_cache_num[pred] + 1
                self.lsh_count += 1

        if not only_test:
            for key in centroid_cache.keys():
                if key in self.data.old_classes:
                    self.data.centroids[key] = 1 * torch.tensor(self.data.centroids[key]).to('cpu') + 0 * (centroid_cache[key] / centroid_cache_num[key])
                else:
                    self.data.centroids[key] = 0.9 * torch.tensor(self.data.centroids[key]).to('cpu') + 0.1 * (centroid_cache[key] / centroid_cache_num[key])
                
            _cen_set = {}
            for cls, cent in self.data.centroids.items():
                _cen_set[cls] = torch.norm(cent,p=2).detach().cpu().numpy().item()
            self.cen_set[self.t] = _cen_set
            self.t +=1

        return preds


   