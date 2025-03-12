# -----------------------------------------------------------------------------
# Functions for SS-KMeans Method
# -----------------------------------------------------------------------------
import os
from tqdm import tqdm
from glob import glob

import torch
import numpy as np
from torch import nn
from sklearn.cluster import KMeans

import util.globals as globals
from util.eval_util import log_accs_from_preds
from util.util import info
from models.sskmeans_utils.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans


# from __future__ import print_function
# import collections
# import argparse
# import os
# import sys
# import math
# import time

# import datetime
# import numpy as np
# import tensorflow as tf
# from copy import deepcopy
# from six.moves import cPickle as pickle
# from tqdm import tqdm

# from utils.data_utils import construct_split_cifar
# from utils.utils import get_sample_weights, sample_from_dataset, update_episodic_memory, concatenate_datasets, samples_for_each_class, sample_from_dataset_icarl, compute_fgt, load_task_specific_data
# from utils.utils import average_acc_stats_across_runs, average_fgt_stats_across_runs, update_reservior, der_update_reservior, average_ltr_across_runs
# from utils.vis_utils import plot_acc_multiple_runs, plot_histogram, snapshot_experiment_meta_data, snapshot_experiment_eval, snapshot_task_labels
# from model import Model
# import os
# from scipy.spatial import distance
# from utils.sv_knn_buffer import SVKNNBuffer
# from utils.buffer import GSS_Buffer
# from utils.buffer import Buffer





device = torch.device('cuda')


def eval_kmeans_semi_sup(args, model, data_loader, stage_i, K=None):
    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """
    all_acc, old_acc, new_acc = None, None, None

    args.ttd_model == 'TTD_L2P_known_K'
    _, original_model = model
    original_model = original_model.cuda()

    # Get the ground truth number of classes for this stage when K is not provided, i.e, when K is known
    if K is None:
        K = int(args.labelled_data + (stage_i * ((args.classes - args.labelled_data) // args.n_stage)))

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to Old classes

    # Load fine-tuned pretrained model
    model = load_finetuned_model(args, model, stage_i)

    # If use pretrained model to evaluate
    if args.use_pretrained_model_for_eval:
        model = use_pretrained_model(args)

    # Extract all features
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Test w/ {args.eval_version} metric', leave=False, bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):
            data, label, _, mask_lab_ = batch
    
            args.ttd_model == 'TTD_L2P_known_K'
            dino_features = original_model(data.cuda())['pre_logits'] 
            feats = model(data.cuda(), task_id=stage_i, cls_features=dino_features)['x'][:, 0] 

            feats = torch.nn.functional.normalize(feats, dim=-1)

            all_feats.append(feats.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(globals.discovered_K) else False for x in label]))
            mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)


    l_feats = all_feats[mask_lab]       # Get labelled set
    u_feats = all_feats[~mask_lab]      # Get unlabelled set
    l_targets = targets[mask_lab]       # Get labelled targets
    u_targets = targets[~mask_lab]      # Get unlabelled targets

    info(f'Fitting Semi-Supervised K-Means stage: {stage_i} ...')
    kmeans = SemiSupKMeans(
        k=K, 
        tolerance=1e-4, 
        max_iterations=args.max_kmeans_iter, 
        init='k-means++',
        n_init=args.k_means_init, 
        random_state=None, 
        n_jobs=None, 
        pairwise_batch_size=1024, 
        mode=None
    )

    l_feats, u_feats, l_targets, u_targets = (
        torch.from_numpy(x).to(device) for x in (l_feats, u_feats, l_targets, u_targets)
    )

    kmeans.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeans.labels_.cpu().numpy()
    u_targets = u_targets.cpu().numpy()

    # -----------------------
    # EVALUATE
    # -----------------------
    # Get preds corresponding to unlabelled set
    preds = all_preds[~mask_lab]

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)

    # -----------------------
    # EVALUATE
    # -----------------------
    if args.test:
        all_acc, old_acc, new_acc = log_accs_from_preds(
            y_true=u_targets, y_pred=preds, mask=mask, 
            eval_funcs=args.eval_funcs,
            save_name=args.save_path, 
            print_output=True, 
            indicator=f"SS-Kmeans_test_stage_{stage_i}_w_{args.eval_version}_metrics",
        )
        info(f'All Acc: {all_acc:.4f} | Old Acc: {old_acc:.4f} | New Acc: {new_acc:.4f}')
    
    # -----------------------
    # SAVE UNLABELLED PREDS
    # -----------------------
    if args.train:
        if args.transductive_evaluation:
            unlabelled_preds_path = os.path.join(args.save_path, 'pred_labels', f'pred_labels_stage_{stage_i}_train.txt')
        else:
            unlabelled_preds_path = os.path.join(args.save_path, 'pred_labels', f'pred_labels_stage_{stage_i}.txt')
        f = open(unlabelled_preds_path, 'w')
        for label in preds:
            f.write(str(label)+"\n")
        
    return all_acc, old_acc, new_acc

def eval_kmeans(args, model, val_loader, stage_i, epoch=None):
    """
    In this case, the test loader only consists of labelled dataset
    """
    args.ttd_model == 'TTD_L2P_known_K'
    model, original_model = model
    model.eval()
    original_model.eval()
    original_model = original_model.cuda() # check if this is necessary

    K = args.labelled_data

    val_feats = []
    val_targets = np.array([])
    mask = np.array([])

    # If use pretrained model to evaluate
    if args.use_pretrained_model_for_eval:
        model = use_pretrained_model(args)

    val_feats = []
    val_targets = []
    mask = []

    for data, label, _, _ in tqdm(val_loader, desc=f'Eval @ epoch: {epoch}', leave=False, bar_format="{desc}{percentage:3.0f}%|{bar}{r_bar}", ncols=80):
        args.ttd_model == 'TTD_L2P_known_K'           
        dino_features = original_model(data.cuda())['pre_logits']
        feats = model(data.cuda(), task_id=stage_i, cls_features=dino_features)['x'][:, 0]

        feats = torch.nn.functional.normalize(feats, dim=-1)

        val_feats.append(feats)
        val_targets.append(label)
        mask.append(np.array([True if x.item() in range(K) else False for x in label]))


    val_feats = torch.cat(val_feats).cpu().numpy()
    val_targets = np.concatenate(val_targets)
    mask = np.concatenate(mask) 

    # -----------------------
    # K-MEANS
    # -----------------------
    # info('Fitting K-Means...')
    kmeans = KMeans(n_clusters=args.classes).fit(val_feats)
    val_preds = kmeans.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc  = log_accs_from_preds(
        y_true=val_targets, y_pred=val_preds, mask=mask,
        eval_funcs=args.eval_funcs, 
        save_name=args.save_path, 
        print_output=True, 
        indicator=f"Kmeans_eval_stage_{stage_i}",
        T=epoch
    )

    return  all_acc, old_acc, new_acc



def use_pretrained_model(args):

    if args.selected_pretrained_model_for_eval == 'dino':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
        info("Use ViT base16 DINO pretrained model")
        state_dict = torch.load(args.dino_pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

    elif args.selected_pretrained_model_for_eval == 'gcd':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
        info("Use ViT base16 GCD pretrained model")
        state_dict = torch.load(args.warmup_model_dir, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    else:
        ValueError('Selected pretrained model {} does not exist.'.format(args.selected_pretrained_model_for_eval))

    model.cuda()

    return model


def load_finetuned_model(args, model, stage_i):
    if args.ttd_model == 'TTD_L2P_known_K':
        model, _ = model
        info(f"Use {args.ttd_model} stage {stage_i} model for testing")
        state_dict = torch.load(glob(os.path.join(args.save_path, 'model', f"{args.ttd_model}_stage_{stage_i}_model_best.pt"))[0], map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        ValueError('Model {} does not exist.'.format(args.ttd_model))

    model.cuda()
    model.eval()

    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.track_running_stats=False

    return model