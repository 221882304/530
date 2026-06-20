# src/post_processing/quality_assessment.py
import numpy as np

class TrajectoryQualityAssessor:
    def __init__(self, initial_th=0.7):
        self.repair_th = initial_th
        self.history = []

    def assess_single(self, traj):
        length_score = min(len(traj['positions']) / 100, 1.0)
        idsw_score = 1.0 - min(traj.get('id_switches', 0) / len(traj['positions']), 1.0)
        vel_var = np.var(traj.get('velocities', []))
        smooth_score = 1.0 / (1.0 + vel_var)
        feat_var = np.var(traj.get('features', []))
        app_score = 1.0 / (1.0 + feat_var)
        return 0.4 * length_score + 0.3 * idsw_score + 0.2 * smooth_score + 0.1 * app_score

    def assess_and_feedback(self, repaired, original):
        if not original:
            return self.repair_th
        success = len(repaired) / len(original)
        self.history.append(success)
        if len(self.history) > 10:
            self.history.pop(0)
        avg = np.mean(self.history)
        if avg > 0.85:
            self.repair_th *= 0.98
        elif avg < 0.7:
            self.repair_th *= 1.02
        return self.repair_th