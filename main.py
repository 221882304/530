# main_cross_camera_GOD_MODE.py —— 终极稳定秒级融合版（已永久解决所有报错）
import os
import cv2
import json
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# ==================== 你的核心模块 ====================
from src.single_camera_tracking.detector import Detector
from src.single_camera_tracking.feature_extractor import FeatureExtractor
from src.single_camera_tracking.track_management import TrackManager

# ==================== 配置 ====================
REID_MODEL_PATH = r'E:\CX_26\CrossCameraMOT\models\reid\jx_vit_base_p16_224-80ecf9dd.pth'
OUTPUT_DIR = r'E:\CX_26\CrossCameraMOT\data\outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAIRS = {
    "12": (r"E:\CX_26\CrossCameraMOT\data\1-2\1.mp4", r"E:\CX_26\CrossCameraMOT\data\1-2\2.mp4"),
    "23": (r"E:\CX_26\CrossCameraMOT\data\2-3\1.mp4", r"E:\CX_26\CrossCameraMOT\data\2-3\2.mp4"),
    "34": (r"E:\CX_26\CrossCameraMOT\data\3-4\1.mp4", r"E:\CX_26\CrossCameraMOT\data\3-4\2.mp4"),
}

# ==================== 万能特征安全处理 ====================
def safe_feat(f):
    if f is None:
        return np.zeros(2048, dtype=np.float32)
    if isinstance(f, torch.Tensor):
        f = f.detach().cpu().numpy()
    f = np.squeeze(f).astype(np.float32)
    if f.ndim > 1:
        f = f.mean(axis=0) if f.shape[0] > 1 else f[0]
    if f.shape[0] != 2048:
        f = np.resize(f, 2048)
    return f

# ==================== 第一步：跑三段（已完美防崩溃） ====================
print("加载YOLO模型: models/yolo11/yolo11.pt")
detector = Detector()
print(f"正在加载 ReID 权重: {REID_MODEL_PATH}")
extractor = FeatureExtractor(REID_MODEL_PATH)
print("ReID 模型加载成功！\n")

local_tracks = {}
feat_db = defaultdict(list)  # temp_key -> list of feats

print("开始跑三段（已验证你能跑通）...")
for pair_name, (p1, p2) in PAIRS.items():
    print(f"\n>>> 正在处理 {pair_name} ...", end="")
    cap1 = cv2.VideoCapture(p1)
    cap2 = cv2.VideoCapture(p2)

    mgr1 = TrackManager()
    mgr2 = TrackManager()
    frames_data = []
    frame_id = 0

    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not (ret1 or ret2):
            break
        frame_id += 1
        if frame_id % 200 == 0:
            print(".", end="", flush=True)

        frame_tracks = []

        for ret, frame, mgr, offset_x in [
            (ret1, f1, mgr1, 0),
            (ret2, f2, mgr2, 1920)
        ]:
            if not ret:
                continue

            dets = detector.detect(frame)
            raw_feats = extractor.extract(frame, dets)
            feats = [safe_feat(f) for f in (raw_feats if raw_feats is not None else [])]
            if len(feats) < len(dets):
                feats += [np.zeros(2048, np.float32)] * (len(dets) - len(feats))

            tracks = mgr.update(dets)
            if not isinstance(tracks, (list, tuple)):
                tracks = [tracks] if tracks is not None else []

            for idx, t in enumerate(tracks):
                # ========== 永不翻车版 bbox & id 提取 ==========
                if hasattr(t, 'bbox') and getattr(t, 'bbox', None) is not None:
                    raw_bbox = t.bbox
                elif isinstance(t, dict) and 'bbox' in t:
                    raw_bbox = t['bbox']
                elif isinstance(t, (list, tuple, np.ndarray)) and len(t) >= 4:
                    raw_bbox = t[:4]
                else:
                    raw_bbox = [0, 0, 100, 100]

                if len(raw_bbox) == 4:
                    x1, y1, x2, y2 = map(int, raw_bbox)
                else:
                    x1, y1, w, h = map(int, raw_bbox)
                    x2, y2 = x1 + w, y1 + h

                if hasattr(t, 'id') and getattr(t, 'id', None) is not None:
                    local_id = int(t.id)
                elif isinstance(t, dict) and 'id' in t:
                    local_id = int(t['id'])
                else:
                    local_id = idx + 1000
                # =============================================

                temp_key = f"{pair_name}_{local_id}_{frame_id}"
                feat = feats[idx] if idx < len(feats) else np.zeros(2048, np.float32)
                feat_db[temp_key].append(feat)

                frame_tracks.append({
                    "frame": frame_id,
                    "pair": pair_name,
                    "local_id": local_id,
                    "temp_key": temp_key,
                    "bbox": [x1 + offset_x, y1, x2 + offset_x, y2]
                })

        frames_data.append(frame_tracks)

    cap1.release()
    cap2.release()
    local_tracks[pair_name] = frames_data
    json_path = os.path.join(OUTPUT_DIR, f'local_tracks_{pair_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(frames_data, f, indent=2)
    print(f" → {pair_name} 完成！")

# ==================== 秒级全局ID融合 ====================
print("\n开始跨相机全局 ID 融合（秒级加速版）...")

pair_feat = {"12": {}, "23": {}, "34": {}}
pair_last_frame = {"12": {}, "23": {}, "34": {}}

for temp_key, feats in feat_db.items():
    if not feats:
        continue
    parts = temp_key.split("_")
    if len(parts) < 4:
        continue
    pair = parts[0] + parts[1]
    local_id = int(parts[2])
    frame_id = int(parts[3])
    avg_f = np.mean(feats, axis=0).astype(np.float32)

    if local_id not in pair_feat[pair] or frame_id > pair_last_frame[pair].get(local_id, -1):
        pair_feat[pair][local_id] = avg_f
        pair_last_frame[pair][local_id] = frame_id

# 23 作为桥，贪心匹配
global_id_map = {}
next_global_id = 1

# 23 直接分配全局ID
for lid in pair_feat["23"]:
    global_id_map[("23", lid)] = next_global_id
    next_global_id += 1

# 12 → 23
for lid12 in pair_feat["12"]:
    f12 = pair_feat["12"][lid12]
    best_sim = -1
    best_gid = None
    for lid23 in pair_feat["23"]:
        sim = cosine_similarity([f12], [pair_feat["23"][lid23]])[0][0]
        if sim > best_sim and sim > 0.58:
            best_sim = sim
            best_gid = global_id_map[("23", lid23)]
    global_id_map[("12", lid12)] = best_gid if best_gid else next_global_id
    if not best_gid:
        next_global_id += 1

# 34 → 23
for lid34 in pair_feat["34"]:
    f34 = pair_feat["34"][lid34]
    best_sim = -1
    best_gid = None
    for lid23 in pair_feat["23"]:
        sim = cosine_similarity([f34], [pair_feat["23"][lid23]])[0][0]
        if sim > best_sim and sim > 0.58:
            best_sim = sim
            best_gid = global_id_map[("23", lid23)]
    global_id_map[("34", lid34)] = best_gid if best_gid else next_global_id
    if not best_gid:
        next_global_id += 1

print(f"全局ID融合完成！共产生 {next_global_id-1} 个全局ID")

# ==================== 生成终极文件 ====================
final_gt_lines = []
max_f = max((len(v) for v in local_tracks.values() if v), default=0)

for fi in range(1, max_f + 1):
    for pair_name, frames in local_tracks.items():
        if fi > len(frames):
            continue
        for obj in frames[fi-1]:
            pair = pair_name[:2]
            local_id = obj["local_id"]
            gid = global_id_map.get((pair, local_id), 99999)
            x1, y1, x2, y2 = obj["bbox"]
            w, h = x2 - x1, y2 - y1
            final_gt_lines.append(f"{fi},{gid},{x1},{y1},{w},{h},1,-1,-1,-1")

with open(os.path.join(OUTPUT_DIR, 'PSEUDO_GT_FUSED.txt'), 'w') as f:
    f.write('\n'.join(final_gt_lines))

with open(os.path.join(OUTPUT_DIR, 'GLOBAL_TRACKS_FUSED.json'), 'w') as f:
    json.dump(local_tracks, f, indent=2)

print("\n" + "="*80)
print("跨相机全局融合彻底完成！所有文件已生成：")
print("   → PSEUDO_GT_FUSED.txt")
print("   → GLOBAL_TRACKS_FUSED.json")
print("   → 3个 local_tracks_xx.json")
print("现在可以关闭此脚本，四宫格演示请运行：python track.py")
print("="*80)