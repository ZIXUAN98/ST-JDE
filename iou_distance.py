import numpy as np
from typing import Union, List, Tuple

def fast_iou_distance_with_filter(
    atracks: Union[List[np.ndarray], List['STrack']],
    btracks: Union[List[np.ndarray], List['STrack']],
    center_thresh: float = 0.25,  # 中心点相对距离阈值
    aspect_thresh: float = 0.5,   # 宽高比差异阈值
    eps: float = 1e-7
) -> np.ndarray:
    """
    带预筛选的高效IoU距离计算
    
    Args:
        atracks: 第一组轨迹或边界框
        btracks: 第二组轨迹或边界框
        center_thresh: 中心点相对距离阈值（0-1之间，越小越严格）
        aspect_thresh: 宽高比差异阈值（0-1之间，越小越严格）
        eps: 防止除零的小量
    
    Returns:
        代价矩阵（1-IoU）
    """
    # 1. 提取边界框坐标
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    
    # 转换输入格式
    if isinstance(atracks[0], np.ndarray):
        a_boxes = np.asarray(atracks, dtype=np.float32)
        b_boxes = np.asarray(btracks, dtype=np.float32)
    else:
        # 假设是STrack对象
        a_boxes = []
        for track in atracks:
            if hasattr(track, 'angle') and track.angle is not None:
                a_boxes.append(track.xywha[:4])  # 旋转框只取前4个坐标
            else:
                a_boxes.append(track.xyxy)
        b_boxes = []
        for track in btracks:
            if hasattr(track, 'angle') and track.angle is not None:
                b_boxes.append(track.xywha[:4])
            else:
                b_boxes.append(track.xyxy)
        a_boxes = np.asarray(a_boxes, dtype=np.float32)
        b_boxes = np.asarray(b_boxes, dtype=np.float32)
    
    n_a, n_b = len(a_boxes), len(b_boxes)
    
    # 2. 计算几何属性（向量化，只计算一次）
    # 中心点坐标
    a_centers = (a_boxes[:, :2] + a_boxes[:, 2:]) / 2
    b_centers = (b_boxes[:, :2] + b_boxes[:, 2:]) / 2
    
    # 宽高
    a_wh = a_boxes[:, 2:] - a_boxes[:, :2]
    b_wh = b_boxes[:, 2:] - b_boxes[:, :2]
    
    # 3. 计算预筛选距离矩阵（完全向量化）
    # 扩展维度用于广播
    a_centers_exp = a_centers[:, np.newaxis, :]  # (n_a, 1, 2)
    b_centers_exp = b_centers[np.newaxis, :, :]  # (1, n_b, 2)
    a_wh_exp = a_wh[:, np.newaxis, :]  # (n_a, 1, 2)
    b_wh_exp = b_wh[np.newaxis, :, :]  # (1, n_b, 2)
    
    # 中心点差值
    center_diff = a_centers_exp - b_centers_exp  # (n_a, n_b, 2)
    
    # 计算两个中心点距离变体（归一化到各自的尺寸）
    # 使用a的尺寸归一化
    center_dist_a = (center_diff[..., 0] / (a_wh_exp[..., 0] + eps))**2 + \
                    (center_diff[..., 1] / (a_wh_exp[..., 1] + eps))**2
    
    # 使用b的尺寸归一化
    center_dist_b = (center_diff[..., 0] / (b_wh_exp[..., 0] + eps))**2 + \
                    (center_diff[..., 1] / (b_wh_exp[..., 1] + eps))**2
    
    # 取两个距离中较小的那个（更宽松的条件）
    center_dist = np.minimum(center_dist_a, center_dist_b)
    
    # 计算宽高比差异
    # 宽高差异（相对差异）
    wh_ratio_a = np.abs(a_wh_exp - b_wh_exp) / (a_wh_exp + eps)
    wh_ratio_b = np.abs(a_wh_exp - b_wh_exp) / (b_wh_exp + eps)
    
    # 宽高比差异（综合两个方向的差异）
    aspect_dist = np.maximum(
        (wh_ratio_a[..., 0] - wh_ratio_a[..., 1])**2,
        (wh_ratio_b[..., 0] - wh_ratio_b[..., 1])**2
    )
    
    # 4. 创建筛选掩码
    # 中心点距离或宽高比差异超过阈值的框对跳过IoU计算
    mask = (center_dist <= center_thresh) & (aspect_dist <= aspect_thresh)
    
    # 5. 初始化代价矩阵（全1表示完全不匹配）
    cost_matrix = np.ones((n_a, n_b), dtype=np.float32)
    
    # 6. 只对通过筛选的框对计算IoU
    if np.any(mask):
        # 获取需要计算IoU的索引
        idx_a, idx_b = np.where(mask)
        
        # 批量计算这些框对的IoU
        if len(idx_a) > 0:
            # 提取对应的框
            a_selected = a_boxes[idx_a]
            b_selected = b_boxes[idx_b]
            
            # 计算IoU（支持4维和5维框）
            if a_boxes.shape[1] == 5 and b_boxes.shape[1] == 5:
                # 旋转框：使用probIoU
                ious = batch_probiou(
                    np.ascontiguousarray(a_selected, dtype=np.float32),
                    np.ascontiguousarray(b_selected, dtype=np.float32),
                ).numpy()
            else:
                # 轴对齐框：计算标准IoU
                ious = compute_batch_iou_optimized(a_selected, b_selected)
            
            # 更新代价矩阵
            cost_matrix[idx_a, idx_b] = 1 - ious
    
    return cost_matrix


def compute_batch_iou_optimized(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    优化的批量IoU计算，只计算指定的框对
    """
    # 向量化计算交集
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1])
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2])
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3])
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4])
    
    # 计算交集面积
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # 计算各自面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 计算并集面积和IoU
    union_area = area1 + area2 - inter_area.flatten()
    iou = inter_area.flatten() / np.maximum(union_area, 1e-8)
    
    return iou


def batch_probiou(rotated_boxes1: np.ndarray, rotated_boxes2: np.ndarray) -> np.ndarray:
    """
    旋转框的概率IoU计算（简化版）
    实际使用时可以替换为更高效的具体实现
    """
    # 这里实现一个简化的版本，实际应用中可以使用更高效的实现
    n1, n2 = len(rotated_boxes1), len(rotated_boxes2)
    ious = np.zeros(min(n1, n2), dtype=np.float32)
    
    # 简化的旋转框IoU计算
    for i in range(min(n1, n2)):
        ious[i] = compute_rotated_iou_simple(
            rotated_boxes1[i], rotated_boxes2[i]
        )
    
    return ious


def compute_rotated_iou_simple(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    简化的旋转框IoU计算
    实际应用中应该使用更精确的方法
    """
    # 这里简化为忽略角度，计算轴对齐IoU
    # 实际应该使用旋转框相交面积计算
    x1 = max(box1[0] - box1[2]/2, box2[0] - box2[2]/2)
    y1 = max(box1[1] - box1[3]/2, box2[1] - box2[3]/2)
    x2 = min(box1[0] + box1[2]/2, box2[0] + box2[2]/2)
    y2 = min(box1[1] + box1[3]/2, box2[1] + box2[3]/2)
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    
    return inter / (area1 + area2 - inter + 1e-8)