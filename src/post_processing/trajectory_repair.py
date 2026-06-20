# src/post_processing/trajectory_repair.py
import numpy as np
from scipy.interpolate import CubicSpline

class TrajectoryRepair:
    def repair(self, tracks_all):
        """
        分级修复轨迹
        tracks_all: List[List[dict]] 每帧一个 list，里面是轨迹 dict
        """
        repaired_all = []
        for frame_tracks in tracks_all:
            repaired_frame = []
            for traj in frame_tracks:  # 每个轨迹 dict
                positions = traj.get('positions', [])  # 假设有 positions 列表
                if not positions:
                    repaired_frame.append(traj)
                    continue

                gap = traj.get('gap', 0)

                if gap < 4:
                    # 线性插值
                    times = np.arange(len(positions))
                    interp = np.interp(times, traj.get('known_times', times), positions, axis=0)
                    traj['positions'] = interp.tolist()
                elif gap < 11:
                    # 样条插值
                    known_times = traj.get('known_times', np.arange(len(positions)))
                    if len(known_times) > 1:
                        cs = CubicSpline(known_times, positions)
                        new_times = np.arange(min(known_times), max(known_times) + 1)
                        traj['positions'] = cs(new_times).tolist()

                repaired_frame.append(traj)

            repaired_all.append(repaired_frame)

        return repaired_all