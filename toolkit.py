import torch
import torch.nn as nn
from collections import Counter
import numpy as np
import torch.nn.functional as F
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

from mydraw.draw import plot, plot2
import itertools
from tqdm import tqdm

def get_neighbors(hash_value, magnitude_range=2, hamming_distance=0):
    """
    Generate local bucket hashes considering magnitude range and Hamming distance.
    
    Args:
        hash_value (str): The original hash value (e.g., "100010").
        magnitude_range (float): The range of magnitude (e.g., 0.1).
        hamming_distance (int): The maximum Hamming distance to consider.
        
    Returns:
        list: A list of neighboring bucket hashes.
    """
    # Extract magnitude and angle part
    magnitude_prefix = hash_value[:2]
    angle_bits = hash_value[2:]
    
    # Step 1: Generate neighboring magnitude prefixes
    current_magnitude = int(magnitude_prefix)
    magnitude_neighbors = []
    for i in [-1, 0, 1]:
        neighbor = current_magnitude + i * magnitude_range
        if 0 <= neighbor <= 1000:
            magnitude_neighbors.append(f"{int(neighbor):02d}")
    
    # Step 2: Generate all combinations of Hamming distance for angle bits
    n_bits = len(angle_bits)
    angle_neighbors = []
    bit_indices = list(range(n_bits))
    
    # Generate all possible combinations with the given Hamming distance
    for flip_indices in combinations(bit_indices, hamming_distance):
        # Flip the bits at the specified indices
        neighbor_bits = list(angle_bits)
        for idx in flip_indices:
            neighbor_bits[idx] = '1' if neighbor_bits[idx] == '0' else '0'
        angle_neighbors.append("".join(neighbor_bits))
    
    # Step 3: Combine magnitude and angle neighbors
    local_hashes = []
    for mag in magnitude_neighbors:
        for angle in angle_neighbors:
            local_hashes.append(mag + angle)
    
    return local_hashes



def real_time_eval(data, record, preds, labels):
    # for each known sample
    known_num, known_acc = 0, 0
    for pred, label in zip(preds, labels):
        if label in data.old_classes:
            known_acc += pred == label
            known_num +=1
            
    known_acc = known_acc/known_num

    # for each unknown class (true)
    truelabel_agreement_ratios, truelabel_entropys = [], []
    for unknown_label in data.new_classes:
        filtered_preds = [a for a, b in zip(preds, labels) if b == unknown_label]
        if len(filtered_preds) == 0:
            continue
        counter = Counter(filtered_preds)
        for cls in data.old_classes:
            counter.pop(cls, None)
        if counter:
            # print(counter)
            most_common_count = counter.most_common(1)[0][1]
            probabilities = [count / len(filtered_preds) for count in counter.values()]
            truelabel_agreement_ratios.append(most_common_count / len(filtered_preds))
        
            probabilities = [count / len(filtered_preds) for count in counter.values()]
            truelabel_entropys.append(-sum(p * np.log2(p) for p in probabilities if p > 0))
        else:
            truelabel_agreement_ratios.append(0)
    
    if len(truelabel_agreement_ratios) > 0 and len(truelabel_entropys) > 0:
        truelabel_agreement_ratio = sum(truelabel_agreement_ratios)/len(truelabel_agreement_ratios)
        truelabel_entropy = sum(truelabel_entropys)/len(truelabel_entropys)
    else:
        truelabel_agreement_ratio = 0
        truelabel_entropy = 0

    # for each unknown class (cluster)
    cluster_agreement_ratios, cluster_entropys = [], []
    for unknown_cluster in range(data.knownclass, data.discovered_class):
        filtered_labels = [a for a, b in zip(labels, preds) if b == unknown_cluster]
        if len(filtered_labels) == 0:
            continue
        counter = Counter(filtered_labels)
        for cls in data.old_classes: 
            counter.pop(cls, None)
        if counter:
            most_common_count = counter.most_common(1)[0][1]
            probabilities = [count / len(filtered_labels) for count in counter.values()]
            cluster_agreement_ratios.append(most_common_count / len(filtered_labels))
            cluster_entropys.append(-sum(p * np.log2(p) for p in probabilities if p > 0))
        else:
            cluster_agreement_ratios.append(0)

    if len(cluster_agreement_ratios) > 0 and len(cluster_entropys) > 0:
        cluster_agreement_ratio = sum(cluster_agreement_ratios)/len(cluster_agreement_ratios)
        cluster_entropy = sum(cluster_entropys)/len(cluster_entropys)
    else:
        cluster_agreement_ratio = 0
        cluster_entropy = 0

    
    cluster_acc = clustering_accuracy(labels.detach().cpu().numpy(), preds)

    ari = adjusted_rand_score(labels.detach().cpu().numpy(), preds)
    
    nmi = normalized_mutual_info_score(labels.detach().cpu().numpy(), preds)
    
    v_measure = v_measure_score(labels.detach().cpu().numpy(), preds)

    # print(f'REAL-TIME EVAL: KA: {known_acc*100:.2f} TA: {truelabel_agreement_ratio*100:.2f} TE: {truelabel_entropy:.2f} CA: {cluster_agreement_ratio*100:.2f} CE: {cluster_entropy:.2f} cluster_acc: {cluster_acc:.2f} ari: {ari:.2f} nmi: {nmi:.2f} v_measure: {v_measure:.2f}')


    # output accumulated eval
    record['input_num'] += len(preds)
    record['a_known_acc'] += known_acc * len(preds)
    record['a_truelabel_agreement_ratio'] += truelabel_agreement_ratio * len(preds)
    record['a_truelabel_entropy'] += truelabel_entropy * len(preds)
    record['a_cluster_agreement_ratio'] += cluster_agreement_ratio * len(preds)
    record['a_cluster_entropy'] += cluster_entropy * len(preds)
    record['a_cluster_acc'] += cluster_acc * len(preds)
    record['a_ari'] += ari * len(preds)
    record['a_nmi'] += nmi * len(preds)
    record['a_v_measure'] += v_measure * len(preds)
    print(f"\nAccumulated EVAL: KA: {record['a_known_acc']*100/record['input_num']:.2f} TA: {record['a_truelabel_agreement_ratio']*100/record['input_num']:.2f} TE: {record['a_truelabel_entropy']/record['input_num']:.2f} CA: {record['a_cluster_agreement_ratio']*100/record['input_num']:.2f} CE: {record['a_cluster_entropy']/record['input_num']:.2f} cluster_acc: {record['a_cluster_acc']/record['input_num']:.2f} ari: {record['a_ari']/record['input_num']:.2f} nmi: {record['a_nmi']/record['input_num']:.2f} v_measure: {record['a_v_measure']/record['input_num']:.2f}")

    return known_acc, truelabel_agreement_ratio, truelabel_entropy, cluster_agreement_ratio, cluster_entropy, cluster_acc, ari, nmi, v_measure
    
def post_eval(ttd, args, data, unlabeled_test_data):
    cluster_labels = {i: [] for i in range(ttd.data.totalclass)}
    ttd.args = args
    ttd.data = data
    print('\n Start Post EVAL...')
    all_preds, all_labels = [], []
    all_input, draw_label = [], []
    for inputs, labels, uq_idxs, mask_lab in tqdm(unlabeled_test_data['default']): 
        inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():

            if ttd.args.ccd_model == 'PromptCCD_w_GMP_known_K' or ttd.args.ccd_model == 'PromptCCD_w_GMP_unknown_K':
                feats = ttd.model(inputs.cuda(), task_id=ttd.stage_i, res=None)['x'][:, 0] 

            elif ttd.args.ccd_model == 'PromptCCD_w_L2P_known_K' or ttd.args.ccd_model == 'PromptCCD_w_DP_known_K':            
                dino_features = ttd.original_model(inputs.cuda())['pre_logits']
                feats = ttd.model(inputs.cuda(), task_id=ttd.stage_i, cls_features=dino_features)['x'][:, 0]

        preds = ttd.predict_and_discover_with_Euclidean_distance(inputs, feats, replay=False, only_test=True, threshold=30)
        # preds = ttd.predict_and_discover_with_magitude(feats,only_test=True)
        # preds = ttd.predict_and_discover_with_cosine_similarity(inputs, feats, replay=False, only_test=True)
        # preds = ttd.predict_and_discover_with_entropy(inputs, feats, replay=False, only_test=True)
        # preds = ttd.predict_and_discover_with_cosine_and_lsh(inputs, feats, replay=False, only_test=True)

        for i in range(len(preds)):
            if preds[i] not in cluster_labels:
                cluster_labels[preds[i]] = []
            cluster_labels[preds[i]].append(labels[i].item())

        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    preds = list(itertools.chain.from_iterable(all_preds))
    labels = list(itertools.chain.from_iterable(all_labels))

    known_num, known_acc = 0, 0
    for pred, label in zip(preds, labels):
        if label in ttd.data.old_classes:
            known_acc += pred == label
            known_num +=1
            
    known_acc = known_acc/known_num

    # for each unknown class (true)
    truelabel_agreement_ratios, truelabel_entropys = [], []
    for unknown_label in ttd.data.new_classes:
        filtered_preds = [a for a, b in zip(preds, labels) if b == unknown_label]
        if len(filtered_preds) == 0:
            continue
        counter = Counter(filtered_preds)
        for cls in ttd.data.old_classes: 
            counter.pop(cls, None)
        if counter:
            most_common_count = counter.most_common(1)[0][1]
            probabilities = [count / len(filtered_preds) for count in counter.values()]
            truelabel_agreement_ratios.append(most_common_count / len(filtered_preds))
            truelabel_entropys.append(-sum(p * np.log2(p) for p in probabilities if p > 0))
        else:
            truelabel_agreement_ratios.append(0)

    if len(truelabel_agreement_ratios) > 0 and len(truelabel_entropys) > 0:
        truelabel_agreement_ratio = sum(truelabel_agreement_ratios)/len(truelabel_agreement_ratios)
        truelabel_entropy = sum(truelabel_entropys)/len(truelabel_entropys)
    else:
        truelabel_agreement_ratio = 0
        truelabel_entropy = 0

    # for each unknown class (cluster)
    cluster_agreement_ratios, cluster_entropys = [], []
    for unknown_cluster in range(ttd.data.knownclass, ttd.data.discovered_class+1):
        filtered_labels = [a.item() for a, b in zip(labels, preds) if b == unknown_cluster]
        if len(filtered_labels) == 0:
            continue
        counter = Counter(filtered_labels)
        for cls in ttd.data.old_classes:
            counter.pop(cls, None)
        if counter:
            most_common_count = counter.most_common(1)[0][1]
            probabilities = [count / len(filtered_labels) for count in counter.values()]
            cluster_agreement_ratios.append(most_common_count / len(filtered_labels))
            cluster_entropys.append(-sum(p * np.log2(p) for p in probabilities if p > 0))
        else:
            cluster_agreement_ratios.append(0)

    if len(cluster_agreement_ratios) > 0 and len(cluster_entropys) > 0:
        cluster_agreement_ratio = sum(cluster_agreement_ratios)/len(cluster_agreement_ratios)
        cluster_entropy = sum(cluster_entropys)/len(cluster_entropys)
    else:
        cluster_agreement_ratio = 0
        cluster_entropy = 0

    cluster_acc = clustering_accuracy(labels, preds)

    ari = adjusted_rand_score(labels, preds)

    nmi = normalized_mutual_info_score(labels, preds)
    
    v_measure = v_measure_score(labels, preds)

    print(f'POST EVAL: KA: {known_acc*100:.2f} TA: {truelabel_agreement_ratio*100:.2f} TE: {truelabel_entropy:.2f} CA: {cluster_agreement_ratio*100:.2f} CE: {cluster_entropy:.2f} cluster_acc: {cluster_acc:.2f} ari: {ari:.2f} nmi: {nmi:.2f} v_measure: {v_measure:.2f}')


class ContrastiveLoss_Eu(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss_Eu, self).__init__()
        self.margin = margin

    def forward(self, feats, labels):
        if isinstance(labels, list):
            labels = torch.tensor(labels, device=feats.device, dtype=torch.long)

        distance_matrix = torch.cdist(feats, feats, p=2)

        labels = labels.unsqueeze(1)
        label_mask = (labels == labels.T).float()
        
        positive_loss = (label_mask * distance_matrix.pow(2)).sum() / (label_mask.sum() + 1e-6)
        
        negative_loss = ((1 - label_mask) * F.relu(self.margin - distance_matrix).pow(2)).sum() / ((1 - label_mask).sum() + 1e-6)
        
        loss = positive_loss + negative_loss
        return loss
    
class ContrastiveLoss_Cos(nn.Module):
    def __init__(self, margin=0.5):

        super(ContrastiveLoss_Cos, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        if isinstance(labels, list):
            labels = torch.tensor(labels, device=embeddings.device, dtype=torch.long)
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        negative_mask = ~positive_mask
        
        positive_loss = (1 - cosine_sim) * positive_mask.float()

        negative_loss = F.relu(cosine_sim - self.margin) * negative_mask.float()
        
        loss = positive_loss.sum() + negative_loss.sum()
        loss /= embeddings.size(0)
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        if isinstance(labels, list):
            labels = torch.tensor(labels, device=embeddings.device, dtype=torch.long)
        dot_product = torch.matmul(embeddings, embeddings.T)

        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        negative_mask = ~positive_mask
        
        positive_loss = (torch.clamp(1 - dot_product, min=0)) * positive_mask.float()

        negative_loss = F.relu(dot_product - self.margin) * negative_mask.float()

        norms = embeddings.norm(p=2, dim=1)
        norm_loss = torch.sum(F.relu(norms - 1))

        loss = positive_loss.sum() + negative_loss.sum()
        loss /= embeddings.size(0) 
        return loss

class DistillationLoss_Centroid(nn.Module):
    def __init__(self, old_classes):
        super(DistillationLoss_Centroid, self).__init__()
        self.old_classes = old_classes


    def forward(self, embeddings, labels, centroids):
        loss = torch.tensor(0.0, requires_grad=True).to(embeddings.device)
        count = 0
        for emb, label in zip(embeddings, labels):
            if label in self.old_classes:
                dis = F.mse_loss(emb, centroids[label].to(embeddings.device))
                loss += dis.detach()
                count +=1 
        if count == 0:
            return loss
        else:
            return loss/count


def clustering_accuracy(all_labels, all_preds):
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    true_classes = np.unique(all_labels)
    pred_classes = np.unique(all_preds)
    
    contingency_matrix = np.zeros((len(true_classes), len(pred_classes)), dtype=np.int64)
    for i, true_label in enumerate(true_classes):
        for j, pred_label in enumerate(pred_classes):
            contingency_matrix[i, j] = np.sum((all_labels == true_label) & (all_preds == pred_label))
    
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    
    accuracy = contingency_matrix[row_ind, col_ind].sum() / len(all_labels)
    return accuracy


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        feat = x
        x = torch.relu(feat)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, feat
