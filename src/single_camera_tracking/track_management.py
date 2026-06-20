# src/single_camera_tracking/track_management.py
def update(self, detections):
    if detections is None:
        detections = []

# src/single_camera_tracking/track_management.py
class TrackManager:
    def __init__(self, max_age=30, min_hits=3):
        self.tracks = []           # 当前活跃轨迹
        self.next_id = 0
        self.max_age = max_age     # 最多消失 30 帧就删除
        self.min_hits = min_hits   # 至少连续出现 3 次才算稳定

    def update(self, detections):
        if not detections:
            # 帧中没检测到 → 所有轨迹年龄+1
            for t in self.tracks:
                t['age'] += 1
            return self.tracks

        # 简单匹配：这里你已经有 associator 了，直接用它的结果
        # 假设 detections 里已经带了 track_id（-1 表示新目标）
        matched_ids = set()
        for det in detections:
            tid = det.get('track_id', -1)
            if tid == -1:
                # 新目标
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': det['bbox'],
                    'age': 0,
                    'hits': 1,
                    'feat': det.get('feat')
                })
                det['track_id'] = self.next_id
                self.next_id += 1
            else:
                # 更新已有轨迹
                for t in self.tracks:
                    if t['id'] == tid:
                        t['bbox'] = det['bbox']
                        t['age'] = 0
                        t['hits'] += 1
                        t['feat'] = det.get('feat')
                        break
                matched_ids.add(tid)

        # 年龄管理：删除太老的轨迹
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

        return self.tracks