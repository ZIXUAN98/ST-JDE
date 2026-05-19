import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import miners

class MetricLearningLoss(nn.Module):
    def __init__(self):
        super(MetricLearningLoss, self).__init__()
        self.mining_func = miners.BatchEasyHardMiner(pos_strategy='hard', neg_strategy='semihard')
        self.loss_func = losses.TripletMarginLoss(margin=0.075)
        # self.margin_scheduler = lambda epoch: 0.075 * (0.95 ** epoch)
        # self.loss_func = CustomTripletLoss(margin=0.075)
        self.confidence_threshold = 1

    def forward(self, embeddings, tags, confidences=None, normalize=False):
        # Select only the embeddings and tags for confidences on top X%
        if confidences is not None and self.confidence_threshold<1:
            top_k = int(self.confidence_threshold * len(confidences))
            _, indices = torch.topk(confidences, top_k, largest=True)
            embeddings = embeddings[indices]
            tags = tags[indices]

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        # Sample triplets and calculate loss
        indices_tuples = self.mining_func(embeddings, tags)
        loss = self.loss_func(embeddings, tags, indices_tuples)
        #loss = self.loss_func(embeddings, tags)
        return loss
    
class CustomTripletLoss(nn.Module):
    def __init__(self, margin=0.075):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, indices_tuple):
        anchors, positives, negatives = indices_tuple
        anchor_emb = embeddings[anchors]
        positive_emb = embeddings[positives]
        negative_emb = embeddings[negatives]

        # 计算归一化距离（这里以gcd_loss思想设计，假设输入是高维向量，先做归一化差异计算）
        # 你需要定义一个适合向量的距离函数，这里示例用欧式距离替代
        def normalized_distance(x1, x2, eps=1e-7):
            diff = x1 - x2
            norm = torch.norm(diff, p=2, dim=1)
            return norm  # 你可以用gcd_loss中的归一化思想替换这里
        
        def normalized_feature_distance(x1, x2, eps=1e-8):
            """
            计算两个批量特征向量 x1 和 x2 之间的归一化欧式距离。

            Args:
                x1, x2: Tensor，形状为 (batch_size, feature_dim)
                eps: 避免除零的小数值

            Returns:
                distances: Tensor，形状为 (batch_size,)，归一化距离
            """
            # L2归一化特征到单位球面
            x1_norm = F.normalize(x1, p=2, dim=1, eps=eps)
            x2_norm = F.normalize(x2, p=2, dim=1, eps=eps)

            # 计算差的欧式距离
            diff = x1_norm - x2_norm
            dist = torch.norm(diff, p=2, dim=1)  # shape: (batch_size,)

            # 归一化距离：除以特征维度的平方根，调整尺度
            feature_dim = x1.shape[1]
            dist_normalized = dist / (feature_dim ** 0.5 + eps)

            return dist_normalized

        pos_dist = normalized_feature_distance(anchor_emb, positive_emb)
        neg_dist = normalized_feature_distance(anchor_emb, negative_emb)

        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()
    
# 1. 动态margin调整
# class AdaptiveMetricLearningLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mining_func = miners.BatchEasyHardMiner(pos_strategy='hard', neg_strategy='semihard')
#         self.base_margin = 0.075
#         # 根据训练进度调整margin
#         self.margin_scheduler = lambda epoch: self.base_margin * (0.95 ** epoch)
    
#     def forward(self, embeddings, tags, confidences=None, epoch=0, normalize=False):
#         current_margin = self.margin_scheduler(epoch)
#         loss_func = losses.TripletMarginLoss(margin=current_margin)
#         if confidences is not None and self.confidence_threshold<1:
#             top_k = int(self.confidence_threshold * len(confidences))
#             _, indices = torch.topk(confidences, top_k, largest=True)
#             embeddings = embeddings[indices]
#             tags = tags[indices]

#         if normalize:
#             embeddings = F.normalize(embeddings, p=2, dim=1)
#         # Sample triplets and calculate loss
#         indices_tuples = self.mining_func(embeddings, tags)
#         loss = self.loss_func(embeddings, tags, indices_tuples)
#         #loss = self.loss_func(embeddings, tags)
#         return loss



