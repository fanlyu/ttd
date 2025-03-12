import torch
import torch.nn as nn
from collections import Counter
import numpy as np
import torch.nn.functional as F
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

import itertools
from tqdm import tqdm


# def get_neighbors(hash_value, norm_range=0.1):
#         # 提取前两位和后四位
#         norm_hash = hash_value[:2]
#         angle_hash = hash_value[2:]
        
#         # 解析特征范数对应的范围
#         norm = int(norm_hash) / 10.0  # 转换为浮点数
#         possible_norms = []
#         for n in range(int((norm - norm_range) * 10), int((norm + norm_range) * 10) + 1):
#             if n <= 10:  # 只需检查上界
#                 possible_norms.append(f"{n:02d}")
        
#         # 计算与后四位汉明距离为 1 的所有组合
#         def generate_hamming_neighbors(binary_str):
#             neighbors = []
#             for i in range(len(binary_str)):
#                 flipped_bit = '1' if binary_str[i] == '0' else '0'
#                 neighbors.append(binary_str[:i] + flipped_bit + binary_str[i+1:])
#             return neighbors
        
#         hamming_neighbors = generate_hamming_neighbors(angle_hash)
        
#         # 拼接所有可能的哈希值
#         neighboring_buckets = []
#         for norm in possible_norms:
#             for neighbor in hamming_neighbors:
#                 neighboring_buckets.append(norm + neighbor)
        
#         return neighboring_buckets


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
    # current_magnitude = int(magnitude_prefix) / 10  # Convert to float (e.g., "10" -> 1.0)
    current_magnitude = int(magnitude_prefix)
    magnitude_neighbors = []
    for i in [-1, 0, 1]:
    # for i in [-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4, -3, -2, -1, 0, 1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
    # for i in [0]:
        neighbor = current_magnitude + i * magnitude_range
        if 0 <= neighbor <= 1000:  # Ensure within valid range [0, 1]
            # magnitude_neighbors.append(f"{int(neighbor * 10):02d}")
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
        filtered_preds = [a for a, b in zip(preds, labels) if b == unknown_label] # 所有标签为unknown_label的样本的预测结果
        if len(filtered_preds) == 0:
            continue
        counter = Counter(filtered_preds)
        # print(counter)
        for cls in data.old_classes: # remove the pred of old class
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
    
    # assert len(truelabel_agreement_ratios) > 0 and len(truelabel_entropys) > 0
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
        for cls in data.old_classes: # remove the pred of old class
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

    # 调整后的兰德指数 (ARI)
    ari = adjusted_rand_score(labels.detach().cpu().numpy(), preds)
    
    # 归一化互信息 (NMI)
    nmi = normalized_mutual_info_score(labels.detach().cpu().numpy(), preds)
    
    # V-Measure
    v_measure = v_measure_score(labels.detach().cpu().numpy(), preds)


    
    # memory_agreements, memory_nums = [], []
    # for unknown_memory in range(data.knownclass, data.discovered_class+1):
    #     # print(unknown_memory)
    #     # print(data.memory_label[unknown_memory])
        
    #     filtered_labels = [a for a in data.memory_label[unknown_memory]]
    #     if len(filtered_labels) == 0:
    #         memory_agreements.append(0)
    #         memory_nums.append(len(data.memory[unknown_memory]))
    #         continue
    #     counter = Counter(filtered_labels)
    #     # for cls in data.old_classes: # remove the pred of old class
    #     #     counter.pop(cls, None)
    #     if counter:
    #         most_common_count = counter.most_common(1)[0][1]
            
    #         memory_agreements.append(most_common_count / len(data.memory[unknown_memory]))
    #         memory_nums.append(len(data.memory[unknown_memory]))
    #     else:
    #         memory_agreements.append(0)
    #         memory_nums.append(len(data.memory[memory_nums]))
    #     # print("class",unknown_memory)
    #     # print("memory_agreements", memory_agreements)
    #     # print("memory_nums",memory_nums)

    # if len(memory_agreements) > 0 and len(cluster_entropys) > 0:
    #     memory_agreement = sum(memory_agreements)/(data.discovered_class+1-data.knownclass)
    #     memory_num = sum(memory_nums)/(data.discovered_class+1-data.knownclass)
    # else:
    #     memory_agreement = 0
    #     memory_num = 0
    
    # # print("#########################")
    # # print("memory_agreements", memory_agreement)
    # # print("memory_nums",memory_num)


    # import os
    # #print(f"Mean Onehot label for cluster {i}: {mean_onehot_label}")
    # f = open(os.path.join(f'memory_label.txt'),'a')
    # print_str = f'{memory_agreement}\n{memory_num}\n'
    # f.write(print_str)
    # f.close()



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
    # print('aux count', ttd.aux_count)
    print('\n Start Post EVAL...')
    all_preds, all_labels = [], []
    all_input, draw_label = [], []
    for inputs, labels, uq_idxs, mask_lab in tqdm(unlabeled_test_data['default']): # Note we discover in all test data
        inputs, labels = inputs.cuda(), labels.cuda() # the labels is not used for output
        # _, feats = self.model.forward(inputs) # do not output this pred

        with torch.no_grad():
            ttd.args.ccd_model == 'TTD_L2P_known_K'
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
    
    # print(cluster_labels)
    from sklearn.preprocessing import OneHotEncoder
    import os
    categories = [list(range(ttd.args.classes))]
    onehot_encoder = OneHotEncoder(categories=categories)
    indicator1 = 'post_cluster_to_label'
    indicator2 = 'label_to_cluster'

    # for i in range(70, len(self.cluster_labels)):
    for i in range(len(cluster_labels)):
        draw_cluster_labels = np.array(cluster_labels[i]).reshape(-1, 1)
        if len(draw_cluster_labels) == 0:
            # 如果聚类没有元素，设置 One-hot 值为全 0
            mean_onehot_label = np.zeros(len(onehot_encoder.categories[0]))  # 确保维度与 One-hot 编码一致
        else:
            onehot_labels = onehot_encoder.fit_transform(draw_cluster_labels)
            mean_onehot_label = np.mean(onehot_labels, axis=0)

        #print(f"Mean Onehot label for cluster {i}: {mean_onehot_label}")
        f = open(os.path.join(f'log_{indicator1}.txt'),'a')
        print_str = f'{mean_onehot_label}'
        f.write(print_str + "\n\n")
        f.close()

        # probabilities = np.squeeze(mean_onehot_label) # ensure the shape of probabilities is (n_classes,)
        # log_probabilities = np.log(np.squeeze(probabilities + 1e-9))  # squeeze the shape of log probabilities as well
        # hot_average_entropy = -np.sum(np.multiply(probabilities, log_probabilities))  # use np.multiply for element-wise multiplication and np.sum for sum
        # print(f"HOT Average entropy for cluster {i}: {hot_average_entropy}")
        # f = open(os.path.join(f'HOT Average entropy for cluster.txt'),'a')
        # print_str = f'HOT Average entropy for cluster {i}: {hot_average_entropy}'
        # f.write(print_str + "\n")
        # f.close()

    preds = list(itertools.chain.from_iterable(all_preds))
    labels = list(itertools.chain.from_iterable(all_labels))


    # for each known sample
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
        # print(counter)
        for cls in ttd.data.old_classes: # remove the pred of old class
            counter.pop(cls, None)
        if counter:
            # print(counter)
            most_common_count = counter.most_common(1)[0][1]
            probabilities = [count / len(filtered_preds) for count in counter.values()]
            truelabel_agreement_ratios.append(most_common_count / len(filtered_preds))
            # print("most_common_count",most_common_count)
            # print(sum(counter.values()))
            # truelabel_agreement_ratios.append(most_common_count / sum(counter.values()))
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
        for cls in ttd.data.old_classes: # remove the pred of old class
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
    # 调整后的兰德指数 (ARI)
    ari = adjusted_rand_score(labels, preds)
    
    # 归一化互信息 (NMI)
    nmi = normalized_mutual_info_score(labels, preds)
    
    # V-Measure
    v_measure = v_measure_score(labels, preds)

    print(f'POST EVAL: KA: {known_acc*100:.2f} TA: {truelabel_agreement_ratio*100:.2f} TE: {truelabel_entropy:.2f} CA: {cluster_agreement_ratio*100:.2f} CE: {cluster_entropy:.2f} cluster_acc: {cluster_acc:.2f} ari: {ari:.2f} nmi: {nmi:.2f} v_measure: {v_measure:.2f}')


class ContrastiveLoss_Eu(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss_Eu, self).__init__()
        self.margin = margin

    def forward(self, feats, labels):
        # 如果 labels 是 list，将其转换为张量
        if isinstance(labels, list):
            labels = torch.tensor(labels, device=feats.device, dtype=torch.long)
        
        # 计算 pairwise 距离矩阵
        distance_matrix = torch.cdist(feats, feats, p=2)  # L2 范数距离
        
        # 构造标签掩码
        labels = labels.unsqueeze(1)
        label_mask = (labels == labels.T).float()  # 相同标签为 1，不同标签为 0
        
        # 正对损失: 距离平方
        positive_loss = (label_mask * distance_matrix.pow(2)).sum() / (label_mask.sum() + 1e-6)
        
        # 负对损失: margin 限制下的距离损失
        negative_loss = ((1 - label_mask) * F.relu(self.margin - distance_matrix).pow(2)).sum() / ((1 - label_mask).sum() + 1e-6)
        
        # 总损失
        loss = positive_loss + negative_loss
        # print(positive_loss, negative_loss)
        return loss
    
class ContrastiveLoss_Cos(nn.Module):
    def __init__(self, margin=0.5):
        """
        初始化对比损失模块
        :param margin: 负对的边界，默认值为 0.5
        """
        super(ContrastiveLoss_Cos, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        前向计算对比损失
        :param embeddings: [batch_size, embedding_dim] 的特征向量
        :param labels: [batch_size] 的标签，正对样本标签相同，负对样本标签不同
        :return: 计算得到的对比损失
        """
        if isinstance(labels, list):
            labels = torch.tensor(labels, device=embeddings.device, dtype=torch.long)
        # 计算余弦相似度矩阵
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        
        # 构建正对和负对的掩码
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [batch_size, batch_size]
        negative_mask = ~positive_mask  # [batch_size, batch_size]
        
        # 正对损失
        positive_loss = (1 - cosine_sim) * positive_mask.float()

        # 负对损失
        negative_loss = F.relu(cosine_sim - self.margin) * negative_mask.float()
        
        # 综合损失，按 batch 大小归一化
        loss = positive_loss.sum() + negative_loss.sum()
        loss /= embeddings.size(0)  # 平均化损失
        
        # print(positive_loss.sum()/embeddings.size(0), negative_loss.sum()/embeddings.size(0))
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        """
        初始化对比损失模块
        :param margin: 负对的边界，默认值为 0.5
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        前向计算对比损失
        :param embeddings: [batch_size, embedding_dim] 的特征向量
        :param labels: [batch_size] 的标签，正对样本标签相同，负对样本标签不同
        :return: 计算得到的对比损失
        """
        if isinstance(labels, list):
            labels = torch.tensor(labels, device=embeddings.device, dtype=torch.long)
        dot_product = torch.matmul(embeddings, embeddings.T)
        
        # 构建正对和负对的掩码
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # [batch_size, batch_size]
        negative_mask = ~positive_mask  # [batch_size, batch_size]
        
        # 正对损失
        positive_loss = (torch.clamp(1 - dot_product, min=0)) * positive_mask.float()

        # 负对损失
        negative_loss = F.relu(dot_product - self.margin) * negative_mask.float()

        norms = embeddings.norm(p=2, dim=1)  # [batch_size]
        norm_loss = torch.sum(F.relu(norms - 1))  # 超过 1 的部分惩罚
        
        # 综合损失，按 batch 大小归一化
        loss = positive_loss.sum() + negative_loss.sum()
        loss /= embeddings.size(0) #+ 10 * norm_loss  # 平均化损失
        
        # print(positive_loss.sum()/embeddings.size(0), negative_loss.sum()/embeddings.size(0))
        return loss

class DistillationLoss_Centroid(nn.Module):
    def __init__(self, old_classes):
        super(DistillationLoss_Centroid, self).__init__()
        self.old_classes = old_classes


    def forward(self, embeddings, labels, centroids):
        """
        前向计算对比损失
        :param embeddings: [batch_size, embedding_dim] 的特征向量
        :param labels: [batch_size] 的标签，正对样本标签相同，负对样本标签不同
        :return: 计算得到的对比损失
        """
        # if isinstance(labels, list):
        #     labels = torch.tensor(labels, device=embeddings.device, dtype=torch.long)

        loss = torch.tensor(0.0, requires_grad=True).to(embeddings.device)
        count = 0
        for emb, label in zip(embeddings, labels):
            if label in self.old_classes:
                # print(f"target_centroid{label}:, type: {type(centroids[label])}")
                dis = F.mse_loss(emb, centroids[label].to(embeddings.device))
                loss += dis.detach()
                count +=1 
        if count == 0:
            return loss
        else:
            return loss/count


def clustering_accuracy(all_labels, all_preds):
    """
    计算聚类准确率 (Clustering Accuracy, CA)，支持预测标签和真实标签值不同的情况。
    
    参数:
    - all_labels: 真实标签列表 (list or numpy array)
    - all_preds: 预测的聚类标签列表 (list or numpy array)
    
    返回:
    - accuracy: 聚类准确率 (float)
    """
    # 将标签转换为 NumPy 数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # 获取唯一的真实标签和预测标签
    true_classes = np.unique(all_labels)
    pred_classes = np.unique(all_preds)
    
    # 构建混淆矩阵
    contingency_matrix = np.zeros((len(true_classes), len(pred_classes)), dtype=np.int64)
    for i, true_label in enumerate(true_classes):
        for j, pred_label in enumerate(pred_classes):
            contingency_matrix[i, j] = np.sum((all_labels == true_label) & (all_preds == pred_label))
    
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)  # 最大化匹配
    
    # 计算聚类准确率
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
        feat = x #/ torch.norm(x, p=2, dim=-1, keepdim=True) #* torch.tanh(1 * torch.norm(x, p=2, dim=-1, keepdim=True))
        x = torch.relu(feat)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, feat
