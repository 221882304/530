# src/cross_camera_association/motion_consistency.py
def motion_cost(track1, track2):
    v1 = track1.get('velocity', 0)
    v2 = track2.get('velocity', 0)
    return abs(v1 - v2) / max(v1, v2, 1e-5)