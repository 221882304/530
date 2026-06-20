# src/preprocessing/spatial_calibrator.py
class SpatialCalibrator:
    def __init__(self, calib):
        self.distance_m = calib['tunnel_distance_m']

    def estimate_distance(self, bbox1, bbox2):
        scale1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        scale2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        return self.distance_m * (1 - scale2 / scale1)