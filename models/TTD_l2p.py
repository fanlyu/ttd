# -----------------------------------------------------------------------------
# PromptTTD model training with Prompt Pool (L2P)
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
from models.sskmeans import eval_kmeans, eval_kmeans_semi_sup

import numpy as np

import os
import pickle
from collections import defaultdict, Counter
import itertools
import random
from toolkit import get_neighbors, SimpleNN, ContrastiveLoss, real_time_eval, post_eval,DistillationLoss_Centroid

import time

from hashmemoryTTD import HashMemory, TTD_simple



device = torch.device('cuda:0')


class TTD_Model:
    def __init__(self, args, model, stage_i):
        super().__init__()
        self.args = args
        self.stage_i = stage_i
        if model == None:
            self.model, self.original_model, self.projection_head = get_vit_model(args)
        else:
            (self.model, self.original_model, self.projection_head) = model
            print(f'Loading best model and projection head state dict from stage {self.stage_i - 1}...')
            self.model.load_state_dict(torch.load(os.path.join(args.save_path, 'model', f'{args.ttd_model}_stage_{self.stage_i - 1}_model_best.pt')))
            self.projection_head.load_state_dict(torch.load(os.path.join(args.save_path, 'model', f'{args.ttd_model}_stage_{self.stage_i - 1}_proj_head_best.pt')))

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
    
    def get_criterion_test(self, feats, preds, centroids, margin=0.5):
        mag_loss = self.mag_constraint(feats, preds)
        contrastive_loss = ContrastiveLoss(margin)(feats, preds)
        distill_loss = DistillationLoss_Centroid(self.old_classes)(feats, preds, centroids)
        loss = 0.01 * contrastive_loss
        loss = (0.1*contrastive_loss + 100*distill_loss)*0.01

        return loss


    def fitobj(self, train_loader, val_loader):
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
                    # we only evaluate on the 'old' classes, to mimic the TTD setting
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
                torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{self.args.ttd_model}_stage_{self.stage_i}_model.pt'))
                torch.save(self.projection_head.state_dict(), os.path.join(self.model_path,f'{self.args.ttd_model}_stage_{self.stage_i}_model_proj_head.pt'))

                if old_acc_test > best_test_acc_lab:
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{self.args.ttd_model}_stage_{self.stage_i}_model_best.pt'))
                    torch.save(self.projection_head.state_dict(), os.path.join(self.model_path, f'{self.args.ttd_model}_stage_{self.stage_i}_proj_head_best.pt'))
                    best_test_acc_lab = old_acc_test

        return self.model, self.original_model, self.projection_head
    

    def eval(self, test_loader):
        # self.args.test = True
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
                        # Extract features with pretrained model
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

            # centroids_filepath = 'centroids_l2p_tiny140.pkl'
            # classes_filepath = 'classes_l2p_tiny140.pkl'
            # centroids_filepath = 'centroids_l2p_cub140.pkl'
            # classes_filepath = 'classes_l2p_cub140.pkl'
            centroids_filepath = 'centroids_l2p_cifar90.pkl'
            classes_filepath = 'classes_l2p_cifar90.pkl'


            with open(centroids_filepath, 'wb') as f:
                pickle.dump(self.centroids_, f)
            with open(classes_filepath, 'wb') as f:
                pickle.dump(self.classes_, f)
            

        return self.model, self.original_model, self.projection_head
            

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


            self.T1 = time.perf_counter()
            TTD = TTD_simple(self.args, self.model, self.original_model, self.projection_head, self, HASH)
 
            HashMemory.initialize_memory(self)

            # TTD.extract_features_and_labels(unlabeled_test_data, self.stage_i)

            TTD.predict_and_discover(unlabeled_test_data, self.stage_i)

            # ----------------
            # LOG
            # ----------------
            torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{self.args.ttd_model}_stage_{self.stage_i}_model.pt'))
            torch.save(self.projection_head.state_dict(), os.path.join(self.model_path,f'{self.args.ttd_model}_stage_{self.stage_i}_model_proj_head.pt'))

            # if old_acc_test > best_test_acc_lab:
            torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{self.args.ttd_model}_stage_{self.stage_i}_model_best.pt'))
            torch.save(self.projection_head.state_dict(), os.path.join(self.model_path, f'{self.args.ttd_model}_stage_{self.stage_i}_proj_head_best.pt'))
                # best_test_acc_lab = old_acc_test

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

        

        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()

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
    # print("labels:", labels)

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
    print("use pretain")
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



