# src/global_tracking/global_track_manager.py (修改后)
import json
import os
from src.utils.config_loader import load_config  # 改为使用 config_loader 加载 YAML

class GlobalTrackManager:
    def __init__(self):
        config = load_config('configs/track_config.yaml')  # 加载 YAML 配置
        self.tracks = []  # 全局轨迹列表
        self.pseudo_label_mode = config.get('pseudo_label_mode', False)
        self.output_dir = 'data/outputs/'

    def manage(self, cross_tracks):  # 新增 manage 方法，调用 update_tracks
        self.update_tracks(cross_tracks, group_name='default')  # 假设 group_name
        return self.tracks  # 返回全局轨迹

    def update_tracks(self, new_associations, group_name):  # 原 update_tracks
        # 原逻辑：更新全局轨迹（融合前后摄像头）
        for assoc in new_associations:
            track_id = assoc['id']
            frame = assoc['frame']
            box = assoc['box']
            vehicle_type = assoc['type']  # ordinary/special 从YOLO
            feature = assoc['feature']  # 从ReID

            # 查找或新建轨迹
            existing_track = next((t for t in self.tracks if t['id'] == track_id), None)
            if existing_track:
                existing_track['frames'].append(frame)
                existing_track['boxes'].append(box)
                existing_track['features'].append(feature)
            else:
                self.tracks.append({
                    'id': track_id,
                    'type': vehicle_type,
                    'frames': [frame],
                    'boxes': [box],
                    'features': [feature]
                })

        if self.pseudo_label_mode:
            output_path = os.path.join(self.output_dir, f'pseudo_trajectories_{group_name}.json')
            with open(output_path, 'w') as f:
                json.dump(self.tracks, f, indent=4)
            print(f"Pseudo trajectories saved to {output_path}")

    # 原其他方法...

# 示例调用（调试用）
if __name__ == "__main__":
    manager = GlobalTrackManager()
    # 模拟关联数据
    associations = [{'id':1, 'frame':0, 'box':[10,20,30,40], 'type':'ordinary', 'feature':[0.1]*768}]
    manager.manage(associations)  # 测试 manage