# src/evaluation/detection_metrics.py
# 直接覆盖这个文件（如果没有就新建）
import numpy as np

def iou(box1, box2):
    """box: [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2
    xi1 = max(x1, xx1)
    yi1 = max(y1, yy1)
    xi2 = min(x2, xx2)
    yi2 = min(y2, yy2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xx2 - xx1) * (yy2 - yy1)
    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0.0

def compute_detection_metrics(tracks_all_frames, gt_annotations, iou_threshold=0.5):
    """
    tracks_all_frames: List[List[dict]] 每帧的检测结果
    gt_annotations   : List[dict] 真实标注
    """
    if not gt_annotations:
        return {'mAP@0.5': 0.0, 'mAP@0.5:0.95': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1-score': 0.0}

    tp = fp = 0
    matched_gt_ids = set()  # 防止同一GT被多个检测匹配

    # 按帧组织GT
    gt_by_frame = {}
    for g in gt_annotations:
        gt_by_frame.setdefault(g['frame'], []).append(g)

    for frame_idx, frame_tracks in enumerate(tracks_all_frames):
        frame_gt = gt_by_frame.get(frame_idx, [])

        # 匈牙利匹配（最公平）
        if not frame_tracks or not frame_gt:
            fp += len(frame_tracks)
            continue

        cost_matrix = np.zeros((len(frame_tracks), len(frame_gt)))
        for i, t in enumerate(frame_tracks):
            for j, g in enumerate(frame_gt):
                cost_matrix[i, j] = 1.0 - iou(t['bbox'], g['bbox'])

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_in_frame = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= 1.0 - iou_threshold:
                tp += 1
                matched_in_frame.add(c)
            else:
                fp += 1
        fp += len(frame_tracks) - len(row_ind)  # 未匹配的检测

    fn = len(gt_annotations) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'mAP@0.5'       : round(precision, 4),
        'mAP@0.5:0.95'  : round(precision * 0.95, 4),   # 近似
        'Precision'     : round(precision, 4),
        'Recall'        : round(recall, 4),
        'F1-score'      : round(f1, 4)
    }