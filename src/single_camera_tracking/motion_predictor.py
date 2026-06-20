# src/single_camera_tracking/motion_predictor.py
import numpy as np

class MotionPredictor:
    def __init__(self):
        # 简单的常速度模型（实际项目可用 KalmanFilter）
        pass

    def predict(self, prev_tracks):
        """
        输入：prev_tracks 可以是 None 或 list
        输出：预测结果，格式和检测框一致
        """
        if prev_tracks is None or len(prev_tracks) == 0:
            return []   # 第一帧或没有轨迹，直接返回空

        predictions = []
        for track in prev_tracks:
            if 'bbox' not in track or len(track.get('history', [])) < 2:
                # 历史太短，无法预测，直接用上一帧位置
                pred_bbox = track['bbox']
            else:
                # 用最近两帧的速度线性外推（常速度模型）
                recent = track['history'][-2:]  # 取最后两个状态
                dx = recent[-1][0] - recent[-2][0]
                dy = recent[-1][1] - recent[-2][1]
                dw = recent[-1][2] - recent[-2][2]
                dh = recent[-1][3] - recent[-2][3]
                # 外推一帧
                x1, y1, x2, y2 = recent[-1]
                pred_bbox = [x1 + dx, y1 + dy, x2 + dw, y2 + dh]
            predictions.append({
                'bbox': pred_bbox,
                'track_id': track.get('id', -1)
            })
        return predictions