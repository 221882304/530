# src/cross_camera_association/appearance_similarity.py
import numpy as np

def app_sim(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))