# -----------------------------------------------------------------------------
# PromptCCD model training with Prompt Pool (L2P)
# -----------------------------------------------------------------------------
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
from models.sskmeans import eval_kmeans, eval_kmeans_semi_sup, eval_kmeans_batch, eval_new_kmeans_batch
from util.eval_util import log_accs_from_preds, log_accs_from_preds_batch, metirc_output
import numpy as np

import os
import pickle
from collections import defaultdict, Counter
import itertools
import random
from toolkit import get_neighbors, real_time_eval, post_eval

device = torch.device('cuda:0')




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
                    dino_features = self.original_model(images.cuda())['pre_logits']
                    feat = self.model(images.cuda(), task_id=self.stage_i, cls_features=dino_features)['x'][:, 0]

            all_feats.append(feat.cpu().numpy()) 
            all_labels.append(labels.cpu().numpy())

        all_feats = np.concatenate(all_feats, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

    def predict_and_discover(self,unlabeled_test_data, stage_i, replay=True, self_correction=True):
        self.stage_i = stage_i
        self.model.eval()

        k = 1
        from collections import deque
        pred_eval = deque(maxlen=10*self.args.batch_size)
        label_eval = deque(maxlen=10)


        # post_eval(self, self.args, self.data, unlabeled_test_data)
        print("############ pre_eval finished ##############")
        batchnum = 1
        for batch in tqdm(unlabeled_test_data['default'], desc='Batches', leave=False, bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):
            inputs, labels, uq_idxs, mask_lab = batch 
            with torch.no_grad():
                dino_features = self.original_model(inputs.cuda())['pre_logits']
                feats = self.model(inputs.cuda(), task_id=self.stage_i, cls_features=dino_features)['x'][:, 0]
            
            feat_cpu = feats.cpu().detach().numpy()
            feat_norms = np.linalg.norm(feat_cpu, axis=1, keepdims=True)
            
            ############ predict and discover novel class ##############
            # preds = self.predict_and_discover_with_Euclidean_distance(inputs, feats, replay=replay)
            # preds = self.predict_and_discover_with_magitude(feats)
            # preds = self.predict_and_discover_with_cosine_similarity(inputs, feats, replay=replay)
            # preds = self.predict_and_discover_with_entropy(inputs, feats, replay)
            batchnum = batchnum + 1
            preds = self.predict_and_discover_with_lsh(inputs, feats, replay=replay)

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
            
            if self_correction:
                if k > 9 and (k % 2 == 0):
                    self.self_memory_correction()

            del inputs, labels, uq_idxs, mask_lab, feats
            torch.cuda.empty_cache()

        ############ post evaluation ####################
        post_eval(self, self.args, self.data, unlabeled_test_data)

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

            # Extract magnitude and angle part
            magnitude_prefix = hash_value[:2]
            angle_bits = hash_value[2:]

            neighbor_buckets_old = []
            neighbor_buckets = []
            angle_buckets = self.data.find_similar_hashes(angle_bits) + [angle_bits]

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

    def predict_and_discover_with_Euclidean_distance(self, inputs, feats, replay=False, only_test=False, threshold=2):
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

    def predict_and_discover_with_magitude(self, feats, only_test = False, threshold=6):
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


    def predict_and_discover_with_cosine_similarity(self, inputs, feats, replay=False, only_test=False, threshold=0.5):
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

    def predict_and_discover_with_lsh(self, inputs, feats, replay=False, only_test=False, lsh_threshold=3.3):
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
                    
                # Extract magnitude and angle part
                magnitude_prefix = hash_value[:2]
                angle_bits = hash_value[2:]
                    
                self.data.update_hash_graph(angle_bits, feat)

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

                # Extract magnitude and angle part
                magnitude_prefix = hash_value[:2]
                angle_bits = hash_value[2:]

                neighbor_buckets_old = []
                neighbor_buckets = []
                angle_buckets = self.data.find_similar_hashes(angle_bits) + [angle_bits]

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
                
                self.data.update_hash_graph(angle_bits, feat)

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

class PromptCCD_Model:
    def __init__(self, args, model, stage_i):
        super().__init__()
        self.args = args
        self.stage_i = stage_i
        if model == None:
            self.model, self.original_model, self.projection_head = get_vit_model(args)
        else:
            (self.model, self.original_model, self.projection_head) = model
            print(f'Loading best model and projection head state dict from stage {self.stage_i - 1}...')
            self.model.load_state_dict(torch.load(os.path.join(args.save_path, 'model', f'{args.ccd_model}_stage_{self.stage_i - 1}_model_best.pt')))
            self.projection_head.load_state_dict(torch.load(os.path.join(args.save_path, 'model', f'{args.ccd_model}_stage_{self.stage_i - 1}_proj_head_best.pt')))

        self.model_path = os.path.join(args.save_path, 'model')

        self.cur_idx = None
        self.prev_idx = None


    def lalign(self, original_features, augmented_features, alpha=2):
        return torch.mean(torch.norm(original_features - augmented_features, dim=1) ** alpha)

    def lunif(self, x, t=2):
        sq_pdist = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), dim=-1) ** 2
        return torch.mean(torch.exp(-sq_pdist / t))

    def objectosphere_loss(self, features, known_classes, sigma=1.0):
        loss = 0.0
        for p in range(len(features)):
            norm = torch.norm(features[p])
            if known_classes[p]: 
                loss += torch.maximum(sigma - norm, torch.tensor(0.0, device=features.device))
            else: 
                loss += norm
        return loss / len(features)

    def mag_constraint(self, features, preds=None):
        minimum = 0.3
        if not preds:
            return ((1.0 - torch.norm(features, p=2, dim=-1))**2).mean()
        
        sigma = []
        max_value = max(self.pred_count.values())
        ratio_dict = {key: value / max_value for key, value in self.pred_count.items()}
        for pred in preds:
            if pred in self.old_classes:
                sigma.append(1.)
            else:
                if pred in ratio_dict.keys():
                    sigma.append(ratio_dict[pred] * (1- minimum) + minimum)
                else:
                    sigma.append(0.9)
            if pred in self.pred_count.keys():
                self.pred_count[pred] += 1
            else:
                self.pred_count[pred] = 0
        sigma = torch.tensor(sigma).cuda()
        return ((sigma - torch.norm(features, p=2, dim=-1))**2).mean()
    
    def compute_criterion(self, pred, label, feat, contrastive_loss):
        mag = self.mag_constraint(feat)
        loss = contrastive_loss + 0.001*mag

        return loss
    
    def fit_mag(self, train_loader, val_loader):
        optimizer = SGD(
            list(self.projection_head.parameters()) + list(self.model.parameters()), 
            lr=self.args.base_lr, 
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs,
            eta_min=self.args.base_lr * 1e-3,
        )

        # Transfer previous learned prompt params to the new prompt
        if self.args.prompt_pool and self.args.shared_prompt_pool:
            if self.stage_i > 0:
                prev_start = (self.stage_i - 1) * self.args.top_k
                prev_end = self.stage_i * self.args.top_k

                cur_start = prev_end
                cur_end = (self.stage_i + 1) * self.args.top_k

                if (prev_end > self.args.pool_size) or (cur_end > self.args.pool_size):
                    pass
                else:
                    self.cur_idx = (slice(cur_start, cur_end))
                    self.prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        self.model.prompt.prompt[self.cur_idx] = self.model.prompt.prompt[self.prev_idx]

        # Transfer previous learned prompt param keys to the new prompt
        if self.args.prompt_pool and self.args.shared_prompt_key:
            if self.stage_i > 0:
                prev_start = (self.stage_i - 1) * self.args.top_k
                prev_end = self.stage_i * self.args.top_k

                cur_start = prev_end
                cur_end = (self.stage_i + 1) * self.args.top_k

                with torch.no_grad():
                    self.model.prompt.prompt_key[self.cur_idx] = self.model.prompt.prompt_key[self.prev_idx]

        sup_con_crit = SupConLoss()
        best_test_acc_lab = 0

        

        for epoch in trange(self.args.stage1_epochs, desc='Epochs', bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):

            loss_record = AverageMeter()
            train_acc_record = AverageMeter()

            self.projection_head.train()
            self.model.train(True)
            self.original_model.eval()

            for batch in tqdm(train_loader['contrast'], desc='Batches', leave=False, bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):

                images, class_labels, uq_idxs, mask_lab = batch
                mask_lab = mask_lab[:, 0]

                class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
                images = torch.cat(images, dim=0).to(device)

                with torch.no_grad():
                    if self.original_model is not None:
                        # Extract features with pretrained model
                        dino_features = self.original_model(images)['pre_logits']
                    else:
                        dino_features = None

                # Extract features with base model
                output = self.model(images, task_id=self.stage_i, cls_features=dino_features, train=True)
                features = output['x'][:, 0] #['pre_logits']
                feat = features

                # Pass features through projection head
                features = self.projection_head(features)

                # L2-normalize features
                features = torch.nn.functional.normalize(features, dim=-1)

                # Choose which instances to run the contrastive loss on
                if self.args.contrast_unlabel_only:
                    # Contrastive loss only on unlabelled instances
                    f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                    con_feats = torch.cat([f1, f2], dim=0)
                else:
                    # Contrastive loss for all examples
                    con_feats = features

                contrastive_logits, contrastive_labels = info_nce_logits(features=con_feats, args=self.args)

                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # Supervised contrastive loss
                f1, f2 = [f[mask_lab] for f in features.chunk(2)]
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                sup_con_labels = class_labels[mask_lab]

                sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

                # # Total loss
                ori_loss = ((1 - self.args.sup_con_weight[self.stage_i]) * contrastive_loss) + (self.args.sup_con_weight[self.stage_i] * sup_con_loss)

                optimizer.zero_grad()
                # # Train acc
                _, pred = contrastive_logits.max(1)
                acc = (pred == contrastive_labels).float().mean().item()

                
                feat_cpu = feat.cpu().detach().numpy()
                feat_norms = np.linalg.norm(feat_cpu, axis=1, keepdims=True)

                loss = self.compute_criterion(pred, class_labels, feat, ori_loss)

                loss.backward()
                optimizer.step()

                train_acc_record.update(acc, pred.size(0))

                loss_record.update(loss.item(), class_labels.size(0))

            # # Step schedule
            exp_lr_scheduler.step()

            if epoch % self.args.eval_every_n_epoch == 0:
                with torch.no_grad():
                    _, old_acc_test, _ = eval_kmeans(
                        args=self.args, 
                        model=(self.model, self.original_model),
                        val_loader=val_loader,
                        stage_i=self.stage_i,
                        epoch=epoch,
                    ) 

                # ----------------
                # LOG
                # ----------------
                torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{self.args.ccd_model}_stage_{self.stage_i}_model.pt'))
                torch.save(self.projection_head.state_dict(), os.path.join(self.model_path,f'{self.args.ccd_model}_stage_{self.stage_i}_model_proj_head.pt'))

                if old_acc_test > best_test_acc_lab:
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{self.args.ccd_model}_stage_{self.stage_i}_model_best.pt'))
                    torch.save(self.projection_head.state_dict(), os.path.join(self.model_path, f'{self.args.ccd_model}_stage_{self.stage_i}_proj_head_best.pt'))
                    best_test_acc_lab = old_acc_test

        return self.model, self.original_model, self.projection_head
    

    def eval(self, test_loader):
        all_acc, old_acc, new_acc = eval_kmeans_semi_sup(
            args=self.args, 
            model=(self.model, self.original_model),
            data_loader=test_loader, 
            stage_i=self.stage_i, 
            K=None,
        )

    def learnCentroid(self, train_loader):
        self.centroids_ = None
        self.classes_ = None
        self.samples_ = {}
        for epoch in trange(self.args.epochs, desc='Epochs', bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):
            for batch in tqdm(train_loader['contrast'], desc='Batches', leave=False, bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):
                images, class_labels, uq_idxs, mask_lab = batch
                mask_lab = mask_lab[:, 0]

                class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
                images = torch.cat(images, dim=0).to(device)

                class_labels = class_labels.cpu().numpy()

                with torch.no_grad():
                    if self.original_model is not None:
                        dino_features = self.original_model(images)['pre_logits']
                    else:
                        dino_features = None

                    output = self.model(images, task_id=self.stage_i, cls_features=dino_features, train=True)
                    features = output['x'][:, 0].cpu().detach().numpy()

                for x, y in zip(features, class_labels):
                    y = int(y.item())
                    if y not in self.samples_:
                        self.samples_[y] = []
                    self.samples_[y].append(x)
                    if len(self.samples_[y]) > 10000:
                        self.samples_[y].pop(0)

            self.classes_ = sorted(self.samples_.keys())
            self.centroids_ = np.zeros((len(self.classes_), features.shape[1]))

            for i, cls in enumerate(self.classes_):
                if len(self.samples_[cls]) > 0:
                    self.centroids_[i] = np.mean(self.samples_[cls], axis=0) 

            centroids_filepath = 'centroids_l2p_cifar70.pkl'
            classes_filepath = 'classes_l2p_cifar70.pkl'


            with open(centroids_filepath, 'wb') as f:
                pickle.dump(self.centroids_, f)
            with open(classes_filepath, 'wb') as f:
                pickle.dump(self.classes_, f)
            
        return self.model, self.original_model, self.projection_head


    def initialize_memory(self):
        self.knownclass = 70 
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

        with open(centroids_filepath, 'rb') as f:
            loaded_centroids = pickle.load(f)
            for k in range(len(loaded_centroids)):
                self.centroids[k] = torch.tensor(loaded_centroids[k], dtype=torch.float32)

        '''
        import numpy as np
        from sklearn.decomposition import PCA

        centroids_matrix = np.stack([self.centroids[k].numpy() for k in range(self.knownclass)], axis=0)

        max_components = centroids_matrix.shape[0] - 1 

        pca = PCA(n_components=self.hash.angle_num)
        pca.fit(centroids_matrix) 

        principal_components = pca.components_  # (angle_num, 768)

        self.hash.hash_vectors = torch.tensor(principal_components, dtype=torch.float32).cpu()
        self.hash.hash_vectors = self.hash.hash_vectors / self.hash.hash_vectors.norm(dim=1, keepdim=True)'
        '''

        for batch in tqdm(self.unlabeled_test_data['default'], desc='Batches', leave=False, bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):
            images, labels, uq_idxs, mask_lab = batch
            with torch.no_grad():
                labels = labels.cpu().numpy()

                if self.args.ccd_model == 'PromptCCD_w_GMP_known_K' or self.args.ccd_model == 'PromptCCD_w_GMP_unknown_K':
                    features = self.model(images.cuda(), task_id=self.stage_i, res=None)['x'][:, 0] 

                elif self.args.ccd_model == 'PromptCCD_w_L2P_known_K' or self.args.ccd_model == 'PromptCCD_w_DP_known_K':            
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
                # Extract magnitude and angle part
                magnitude_prefix = hash_value[:2]
                angle_bits = hash_value[2:]

                self.update_hash_graph(angle_bits, features[idx])

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
                    sim = self.cos_sim(feature, self.hash_centroids[h][0])
                    self.sim_graph[hash_val][h] = sim
                    self.sim_graph[h][hash_val] = sim
        else:
            old_centroid, count = self.hash_centroids[hash_val]
            new_centroid = (old_centroid * count + feature) / (count + 1)
            self.hash_centroids[hash_val] = (new_centroid, count + 1)
            
            for h in self.hash_centroids:
                cnt = cnt + 1
                if h != hash_val:
                    sim = self.cos_sim(new_centroid, self.hash_centroids[h][0])
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
        
        

    def TTT(self, unlabeled_test_data, HASH):
        self.unlabeled_test_data = unlabeled_test_data
        self.hash = HASH

        optimizer = SGD(
            list(self.projection_head.parameters()) + list(self.model.parameters()), 
            lr=self.args.base_lr, 
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs,
            eta_min=self.args.base_lr * 1e-3,
        )

        # Transfer previous learned prompt params to the new prompt
        if self.args.prompt_pool and self.args.shared_prompt_pool:
            if self.stage_i > 0:
                prev_start = (self.stage_i - 1) * self.args.top_k
                prev_end = self.stage_i * self.args.top_k

                cur_start = prev_end
                cur_end = (self.stage_i + 1) * self.args.top_k

                if (prev_end > self.args.pool_size) or (cur_end > self.args.pool_size):
                    pass
                else:
                    self.cur_idx = (slice(cur_start, cur_end))
                    self.prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        self.model.prompt.prompt[self.cur_idx] = self.model.prompt.prompt[self.prev_idx]

        # Transfer previous learned prompt param keys to the new prompt
        if self.args.prompt_pool and self.args.shared_prompt_key:
            if self.stage_i > 0:
                prev_start = (self.stage_i - 1) * self.args.top_k
                prev_end = self.stage_i * self.args.top_k

                cur_start = prev_end
                cur_end = (self.stage_i + 1) * self.args.top_k

                with torch.no_grad():
                    self.model.prompt.prompt_key[self.cur_idx] = self.model.prompt.prompt_key[self.prev_idx]

        
        for epoch in trange(self.args.epochs, desc='Epochs', bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):

            self.projection_head.train()
            self.model.train(True)
            self.original_model.eval()

            TTD = TTD_simple(self.args, self.model, self.original_model, self.projection_head, self, HASH)
    
            self.initialize_memory()

            TTD.predict_and_discover(unlabeled_test_data, self.stage_i)
            
            # ----------------
            # LOG
            # ----------------
            torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{self.args.ccd_model}_stage_{self.stage_i}_model.pt'))
            torch.save(self.projection_head.state_dict(), os.path.join(self.model_path,f'{self.args.ccd_model}_stage_{self.stage_i}_model_proj_head.pt'))


            torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{self.args.ccd_model}_stage_{self.stage_i}_model_best.pt'))
            torch.save(self.projection_head.state_dict(), os.path.join(self.model_path, f'{self.args.ccd_model}_stage_{self.stage_i}_proj_head_best.pt'))


        return self.model, self.original_model, self.projection_head
    

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        if (anchor_dot_contrast.size(1) > 0):
            
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
        else:
            logits = anchor_dot_contrast

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def info_nce_logits(features, args):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels


def get_vit_model(args):
    args.interpolation = 3
    args.crop_pct = 0.875

    original_model = vit_base_patch16_224_dino(
        pretrained=True, 
        num_classes=0, 
        prompt_length=args.prompt_length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.pool_size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )

    model = vit_base_patch16_224_dino(
        pretrained=True, 
        num_classes=0, 
        prompt_length=args.prompt_length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.pool_size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )

    original_model.to(device)
    model.to(device)

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in original_model.parameters():
        m.requires_grad = False

    for n, p in model.named_parameters():
        if n.startswith(tuple(args.freeze)):
            p.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)

    projection_head.to(device)

    return model, original_model, projection_head



