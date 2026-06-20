# src/preprocessing/temporal_synchronizer.py
class TemporalSynchronizer:
    def __init__(self, calib):
        self.offset = calib['cam2']['offset_sec']

    def synchronize(self, frames1, frames2, fps=30):
        offset_frames = int(self.offset * fps)
        if offset_frames > 0:
            frames1 = frames1[offset_frames:]
        else:
            frames2 = frames2[-offset_frames:]
        min_len = min(len(frames1), len(frames2))
        return frames1[:min_len], frames2[:min_len]