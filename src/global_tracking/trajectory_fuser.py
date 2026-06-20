# src/global_tracking/trajectory_fuser.py
class TrajectoryFuser:
    def fuse(self, global_tracks):
        fused = []
        # 假设global_tracks是匹配对列表
        for t1, t2 in global_tracks:
            combined = t1.get('positions', []) + t2.get('positions', [])
            fused.append({'id': t1['global_id'], 'positions': combined})
        return fused