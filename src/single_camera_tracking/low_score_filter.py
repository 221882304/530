# src/single_camera_tracking/low_score_filter.py
import numpy as np

class TunnelLowScoreFilter:
    def __init__(self, score_thresh=0.25, iou_thresh=0.3):
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh

    def iou(self, box1, box2):
        """计算两个框的 IoU"""
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
        return inter / union if union > 0 else 0

    def filter(self, detections, prev_tracks):
        """
        核心功能：保留置信度低的检测框，只要它和上一帧的轨迹高度重合
        prev_tracks: list[dict] 或 None
        """
        if not detections:
            return []

        if prev_tracks is None or len(prev_tracks) == 0:
            # 没有历史轨迹 → 只保留高置信度检测
            return [d for d in detections if d['conf'] >= 0.4]

        # 提取上一帧所有轨迹的 bbox
        prev_boxes = []
        for trk in prev_tracks:
            if 'bbox' in trk:
                prev_boxes.append(trk['bbox'])

        filtered_dets = []
        for det in detections:
            box = det['bbox']
            conf = det['conf']

            # 高置信度直接保留
            if conf >= 0.4:
                filtered_dets.append(det)
                continue

            # 低置信度：看是否和上一帧某个轨迹高度重合
            matched = False
            for prev_box in prev_boxes:
                if self.iou(box, prev_box) > self.iou_thresh:
                    matched = True
                    break

            if matched:
                # 强制提升置信度（可选）
                det['conf'] = min(0.6, conf + 0.3)
                filtered_dets.append(det)
            # else: 丢弃

        return filtered_dets