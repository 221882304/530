
import numpy as np

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    px1, py1, px2, py2 = box2
    inter_x1 = max(x1, px1)
    inter_y1 = max(y1, py1)
    inter_x2 = min(x2, px2)
    inter_y2 = min(y2, py2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (px2 - px1) * (py2 - py1)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

def compute_tracking_metrics(tracks, gt, iou_th=0.5):
    """
    计算跟踪指标：MOTA, MOTP, HOTA, IDs, FP, FN
    tracks: List[List[dict]] 每帧的轨迹列表
    gt: List[dict] 标注列表
    """
    if not gt:
        return {'MOTA': 0.0, 'MOTP': 0.0, 'HOTA': 0.0, 'IDs': 0, 'FP': 0, 'FN': 0}

    # GT 每帧列表
    gt_per_frame = {}
    for ann in gt:
        f = ann['frame']
        gt_per_frame.setdefault(f, []).append(ann)

    # Tracks 每帧列表
    track_per_frame = {}
    for f, frame_tracks in enumerate(tracks):
        track_per_frame.setdefault(f, []).extend(frame_tracks)

    fp = fn = ids = 0
    matched_iou_sum = 0.0
    matched_count = 0
    prev_match = {}

    for f in range(len(tracks)):
        gts = gt_per_frame.get(f, [])
        trks = track_per_frame.get(f, [])

        # 构建成本矩阵
        cost = np.zeros((len(trks), len(gts)))
        for i, t in enumerate(trks):
            for j, g in enumerate(gts):
                cost[i, j] = 1 - iou(t['bbox'], g['bbox'])

        # 匹配
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(cost)
        for r, c in zip(row, col):
            if cost[r, c] < 1 - iou_th:
                matched_iou_sum += 1 - cost[r, c]
                matched_count += 1
                # ID 切换检测
                tr_id = trks[r].get('global_id', trks[r].get('id', -1))
                gt_id = gts[c]['id']
                if tr_id in prev_match and prev_match[tr_id] != gt_id:
                    ids += 1
                prev_match[tr_id] = gt_id

        fp += len(trks) - matched_count
        fn += len(gts) - matched_count

    total_gt = len(gt)
    mota = 1 - (fp + fn + ids) / total_gt if total_gt > 0 else 0.0
    motp = matched_iou_sum / matched_count if matched_count > 0 else 0.0
    hota = mota * motp  # 简化

    return {
        'MOTA': round(mota, 4),
        'MOTP': round(motp, 4),
        'HOTA': round(hota, 4),
        'IDs': ids,
        'FP': fp,
        'FN': fn
    }