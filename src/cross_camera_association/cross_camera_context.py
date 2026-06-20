# src/cross_camera_association/cross_camera_context.py
def context_score(frame1, frame2):
    density1 = len(frame1.get('dets', []))
    density2 = len(frame2.get('dets', []))
    if max(density1, density2) == 0:
        return 0.0
    return min(density1, density2) / max(density1, density2)