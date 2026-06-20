# src/single_camera_tracking/data_association.py
import numpy as np
from scipy.spatial.distance import cdist

class DataAssociation:
    def __init__(self, iou_threshold=0.3, feat_threshold=0.6):
        self.iou_threshold = iou_threshold
        self.feat_threshold = feat_threshold

    def iou_cost(self, box1, box2):
        """计算两个框的 IoU，越大越相似 → cost 越小"""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_p - x1_p) * (y2_p - y1_p)
        union = area1 + area2 - inter
        iou = inter / union if union > 0 else 0
        return 1 - iou  # cost = 1 - iou

    def associate(self, detections, predictions, features=None):
        """
        兼容两种调用方式：
        1. 有 ReID 特征：associate(dets, preds, feats)
        2. 无 ReID 特征：associate(dets, preds)
        """
        if not detections:
            return []  # 没检测到东西，直接返回空

        # 预测为空时，初始化为全0
        if not predictions:
            predictions = [{'bbox': [0,0,1,1]}] * len(detections)

        # 构建成本矩阵
        n_det = len(detections)
        n_pred = len(predictions)
        cost_matrix = np.zeros((n_det, n_pred))

        # 外观 + 运动综合成本
        for i, det in enumerate(detections):
            for j, pred in enumerate(predictions):
                iou_c = self.iou_cost(det['bbox'], pred['bbox'])

                # 如果有 ReID 特征，再加外观成本
                if features is not None and len(features) > i and features[i] is not None:
                    # 假设 predictions[j] 有历史特征
                    if 'feat' in predictions[j] and predictions[j]['feat'] is not None:
                        feat_dist = cdist(features[i].cpu().numpy().reshape(1, -1),
                                         predictions[j]['feat'].cpu().numpy().reshape(1, -1), 'cosine')[0][0]
                        app_cost = feat_dist
                    else:
                        app_cost = 0.5
                    total_cost = 0.6 * iou_c + 0.4 * app_cost
                else:
                    total_cost = iou_c

                cost_matrix[i, j] = total_cost

        # 匈牙利匹配（简单版）
        matched = []
        if n_det > 0 and n_pred > 0:
            row_ind, col_ind = np.where(cost_matrix < 0.7)  # 阈值可调
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.7:
                    matched.append({
                        'det_idx': int(r),
                        'pred_idx': int(c),
                        'cost': float(cost_matrix[r, c])
                    })

        # 构造最终输出（保持你原来的格式）
        results = []
        for m in matched:
            det = detections[m['det_idx']]
            det['track_id'] = predictions[m['pred_idx']].get('track_id', -1)
            results.append(det)

        # 未匹配的当作新目标
        matched_det_idx = {m['det_idx'] for m in matched}
        for i, det in enumerate(detections):
            if i not in matched_det_idx:
                det['track_id'] = -1  # 新目标
                results.append(det)

        return results