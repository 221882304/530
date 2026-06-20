# src/post_processing/trajectory_smoother.py
import numpy as np

class TrajectorySmoother:
    def smooth(self, trajs):
        smoothed = []
        for traj in trajs:
            pos = traj['positions']
            smoothed.append(np.convolve(pos, np.ones(5)/5, mode='valid'))
        return smoothed