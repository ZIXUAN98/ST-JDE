import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings

def gcd_loss(pred, target, eps=1e-7, mode='exp', gamma=1):
    center1 = (pred[..., :2] + pred[..., 2:]) / 2
    center2 = (target[..., :2] + target[..., 2:]) / 2

    whs = center1[..., :2] - center2[..., :2]

    w1 = pred[..., 2] - pred[..., 0]
    h1 = pred[..., 3] - pred[..., 1]
    w2 = target[..., 2] - target[..., 0]
    h2 = target[..., 3] - target[..., 1]

    center_distance1 = (whs[..., 0] / (w1 + eps)) ** 2 + (whs[..., 1] / (h1 + eps)) ** 2
    wh_distance1 = (((w1 - w2) / (w2 + eps)) ** 2 + ((h1 - h2) / (h2 + eps)) ** 2) / 4

    center_distance2 = (whs[..., 0] / (w2 + eps)) ** 2 + (whs[..., 1] / (h2 + eps)) ** 2
    wh_distance2 = (((w1 - w2) / (w1 + eps)) ** 2 + ((h1 - h2) / (h1 + eps)) ** 2) / 4

    gcd_2 = ( center_distance1 + wh_distance1 + center_distance2 + wh_distance2 ) / 2 
    
    if mode == 'exp':
        gcd = torch.exp(-torch.sqrt(gcd_2))
    
    if mode == 'sqrt':
        gcd = torch.sqrt(gcd_2)
    
    if mode == 'log':
        gcd = torch.log(gcd_2 + 1)

    if mode == 'norm_sqrt':
        gcd = 1 - 1 / (gamma + torch.sqrt(gcd_2))

    if mode == 'w2':
        gcd = gcd_2

    return gcd
def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, mpdiou_hw=None):
    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    gcd = gcd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
    gcd_l = ((1.0 - gcd) * weight).sum() / target_scores_sum * 0.0002
    loss_iou = self.iou_ratio * loss_iou +  (1 - self.iou_ratio) * gcd_l