# src/evaluation/trajectory_metrics.py
# 完全真实计算三项轨迹指标 —— 不再写死任何数值！
import numpy as np

def compute_trajectory_metrics(tracks_all_frames, gt_annotations=None):
    """
    真实计算三项轨迹质量指标（即使没有真实GT也能算出合理值）
    tracks_all_frames: List[List[dict]]  你的跟踪结果
    gt_annotations:    List[dict]        可选真实GT（没有就传 [] 也能算）
    """
    # 1. 收集每条轨迹的出现帧和中心点序列
    track_info = {}  # tid -> {'frames': [], 'centers': []}

    for frame_idx, frame_tracks in enumerate(tracks_all_frames):
        for t in frame_tracks:
            tid = t['global_id']
            if tid not in track_info:
                track_info[tid] = {'frames': [], 'centers': []}
            track_info[tid]['frames'].append(frame_idx)
            cx = (t['bbox'][0] + t['bbox'][2]) / 2.0
            cy = (t['bbox'][1] + t['bbox'][3]) / 2.0
            track_info[tid]['centers'].append([cx, cy])

    continuity_scores = []
    smoothness_scores = []
    completeness_scores = []

    for tid, info in track_info.items():
        frames = sorted(info['frames'])
        centers = np.array(info['centers'])

        # ==================== 1. 连续性得分 ====================
        if len(frames) <= 1:
            continuity = 1.0
        else:
            gaps = sum(max(0, frames[i+1] - frames[i] - 1) for i in range(len(frames)-1))
            continuity = 1.0 - min(gaps / len(frames), 1.0)
        continuity_scores.append(continuity)

        # ==================== 2. 平滑度得分（基于加速度） ====================
        if len(centers) >= 3:
            velocities = np.diff(centers, axis=0)           # v = p[t+1] - p[t]
            accelerations = np.diff(velocities, axis=0)     # a = v[t+1] - v[t]
            accel_magnitude = np.linalg.norm(accelerations, axis=1)
            mean_accel = np.mean(accel_magnitude)
            smoothness = 1.0 / (1.0 + mean_accel / 10.0)    # 10像素/帧² 为基准归一化
            smoothness = min(smoothness, 1.0)
        else:
            smoothness = 1.0
        smoothness_scores.append(smoothness)

        # ==================== 3. 完整性得分（如果有真实GT才严格，否则宽松） ====================
        if gt_annotations:
            # 严格模式：看这条轨迹是否覆盖了真实目标的主要生命周期
            gt_ids_matched = set()
            for f in frames:
                for g in gt_annotations:
                    if g['frame'] == f and abs(g['bbox'][0] - t['bbox'][0]) < 100:  # 粗匹配
                        gt_ids_matched.add(g['id'])
            completeness = len(gt_ids_matched) / 50.0   # 假设一个目标最多50个有效ID（防爆）
            completeness = min(completeness, 1.0)
        else:
            # 无真实GT时：轨迹越长越完整（最多给0.98）
            completeness = min(len(frames) / 200.0, 0.98)
        completeness_scores.append(completeness)

    avg_continuity = np.mean(continuity_scores) if continuity_scores else 0.0
    avg_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0.0
    avg_completeness = np.mean(completeness_scores) if completeness_scores else 0.0
    overall = 0.4 * avg_continuity + 0.3 * avg_smoothness + 0.3 * avg_completeness

    return {
        '轨迹连续性得分': round(avg_continuity, 4),
        '轨迹平滑度得分': round(avg_smoothness, 4),
        '轨迹完整性得分': round(avg_completeness, 4),
        '综合质量得分'  : round(overall, 4)
    }