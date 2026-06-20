# src/visualization/video_visualizer.py
import cv2
import os
import numpy as np

class VideoVisualizer:
    def __init__(self, fps=30, resolution=(1920, 1080)):
        self.fps = fps
        self.resolution = resolution
        self.writer = None

    def _init_writer(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(save_path, fourcc, self.fps, self.resolution)

    def visualize(self, tracks, frames, save_path='data/outputs/tracked_video.mp4', mode='w'):
        """
        tracks: list[dict]  轨迹
        frames: list[np.ndarray]  当前帧（可以是1张或2张拼接）
        save_path: 输出路径
        mode: 'w' 重新写入（默认） / 'a' 追加写入
        """
        if not frames:
            return

        # 自动拼接前后摄像头画面（左右拼接）
        if len(frames) == 2:
            frame = np.hstack([frames[0], frames[1]])
        else:
            frame = frames[0]

        # 画轨迹框和ID
        for trk in tracks:
            if 'bbox' not in trk:
                continue
            x1, y1, x2, y2 = map(int, trk['bbox'])
            track_id = trk.get('id', -1)
            color = (0, 255, 0) if track_id != -1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'ID:{track_id}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 初始化或追加写入
        if mode == 'a' and self.writer is None:
            self._init_writer(save_path)

        if self.writer is None:
            self._init_writer(save_path)

        frame = cv2.resize(frame, self.resolution)
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None