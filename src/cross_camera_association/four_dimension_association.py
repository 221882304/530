# src/cross_camera_association/four_dimension_association.py
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .motion_consistency import motion_cost
from .appearance_similarity import app_sim
from .spatio_temporal import spatio_cost
from .cross_camera_context import context_score
from src.utils.logger import setup_logger  # 修改为导入 setup_logger 函数

def estimate_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() / 255.0

def count_objects_per_m2(frame, detections):
    h, w = frame.shape[:2]
    return len(detections) / (h * w / 1000000.0)

def dynamic_fusion_weights(frame1, frame2, dets1, dets2):
    light1 = estimate_brightness(frame1)
    light2 = estimate_brightness(frame2)
    density = max(count_objects_per_m2(frame1, dets1), count_objects_per_m2(frame2, dets2))
    w_app = 0.3 + 0.6 * min(1-light1, 1-light2)
    w_motion = 0.5 - 0.3 * min(density / 10.0, 1.0)
    w_spatio = 0.15
    w_state = 0.1
    total = w_app + w_motion + w_spatio + w_state
    return {'appearance': w_app/total, 'motion': w_motion/total, 'spatio_temporal': w_spatio/total, 'state': w_state/total}

def compute_total_cost(cost_app, cost_motion, cost_st, cost_state, weights):
    return weights['appearance'] * cost_app + weights['motion'] * cost_motion + weights['spatio_temporal'] * cost_st + weights['state'] * cost_state

class FourDimensionAssociation:
    def __init__(self, max_cost=0.5):
        self.max_cost = max_cost
        self.logger = setup_logger('four_dimension_association')  # 初始化 logger

    def associate(self, tracks1, tracks2, frames1, frames2):
        if not tracks1 or not tracks2:
            self.logger.warning("No tracks provided for association. Returning empty matches.")
            return []

        cost_matrix = np.full((len(tracks1), len(tracks2)), np.inf)  # 默认inf，避免无效匹配
        valid_pairs = 0
        for i, t1 in enumerate(tracks1):
            if t1 is None or not isinstance(t1, dict) or 'feat' not in t1 or 'time' not in t1:
                self.logger.warning(f"Skipping invalid track1 at index {i}: {t1}")
                continue
            for j, t2 in enumerate(tracks2):
                if t2 is None or not isinstance(t2, dict) or 'feat' not in t2 or 'time' not in t2:
                    self.logger.warning(f"Skipping invalid track2 at index {j}: {t2}")
                    continue
                try:
                    cost_app = 1 - app_sim(t1['feat'], t2['feat'])
                    cost_motion = motion_cost(t1, t2)
                    cost_st = spatio_cost(t1['time'], t2['time'], {'max_speed_kmh': 200})
                    cost_state = 1 - context_score(frames1[i], frames2[j])
                    weights = dynamic_fusion_weights(frames1[i], frames2[j], tracks1, tracks2)
                    total = compute_total_cost(cost_app, cost_motion, cost_st, cost_state, weights)
                    cost_matrix[i, j] = total if not np.isnan(total) else np.inf  # 处理NaN
                    valid_pairs += 1
                except Exception as e:
                    self.logger.error(f"Error computing cost for track1 {i} and track2 {j}: {str(e)}")
                    continue

        if valid_pairs == 0:
            self.logger.warning("No valid pairs found for association. Returning empty matches.")
            return []

        # 替换inf为大数，避免infeasible
        cost_matrix[np.isinf(cost_matrix)] = 1e9  # 大数代替inf

        row, col = linear_sum_assignment(cost_matrix)
        matches = [(tracks1[r], tracks2[c]) for r, c in zip(row, col) if cost_matrix[r,c] < self.max_cost]
        return matches