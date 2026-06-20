# src/evaluation/overall_metrics.py
import time

def compute_overall_metrics(start_time, tracks, gt):
    run_time = (time.time() - start_time) * 1000  # ms
    precision = len(tracks) / len(gt) if gt else 0  # 简化
    recall = len(tracks) / len(gt) if gt else 0
    accuracy = precision  # 简化
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'Run time (ms)': run_time, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy, 'F1-score': f1}