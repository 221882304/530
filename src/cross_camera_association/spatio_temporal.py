# src/cross_camera_association/spatio_temporal.py
def spatio_cost(t1, t2, calib):
    time_gap = t2 - t1
    dist = calib.get('tunnel_distance_m', 500)
    speed = dist / time_gap * 3.6 if time_gap > 0 else 0
    if speed > calib.get('max_speed_kmh', 200) or time_gap < 2 or time_gap > 10:
        return 1.0
    return 0.0