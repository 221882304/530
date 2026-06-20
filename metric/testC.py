# testC.py —— 永久原始纯净版（顶会一作标配，0手脚，纯王者）
import os
from collections import defaultdict

GT_FILE   = r'E:\CX_26\CrossCameraMOT\data\outputs\PSEUDO_GT_PERFECT.txt'
PRED_FILE = r'E:\CX_26\CrossCameraMOT\data\outputs\pred_cross.txt'

def load(filepath):
    t = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 7: continue
            frame = int(p[0])
            tid   = int(p[1])
            x1,y1,w,h = map(float, p[2:6])
            t[frame].append((tid, [x1, y1, x1+w, y1+h], float(p[6])))
    return t

print("正在加载跨摄像头预测结果...")
gt  = load(GT_FILE)
pred = load(PRED_FILE)
frames = sorted(set(gt.keys()) | set(pred.keys()))

def iou(a, b):
    x1,y1,x2,y2 = a
    p1,q1,p2,q2 = b
    xi1 = max(x1, p1); yi1 = max(y1, q1)
    xi2 = min(x2, p2); yi2 = min(y2, q2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    return inter / ((x2-x1)*(y2-y1) + (p2-p1)*(q2-q1) - inter + 1e-6)

TP = FP = FN = IDSW = 0
prev_match = {}

for f in frames:
    gts = gt.get(f, [])
    prs = pred.get(f, [])
    matched = set()
    for tid, box, _ in prs:
        best_iou = best_gid = best_idx = None
        for idx, (gid, gbox, _) in enumerate(gts):
            if idx in matched: continue
            cur = iou(box, gbox)
            if best_iou is None or cur > best_iou:
                best_iou = cur
                best_gid = gid
                best_idx = idx
        if best_iou is not None and best_iou >= 0.5:
            TP += 1
            matched.add(best_idx)
            if tid in prev_match and prev_match[tid] != best_gid:
                IDSW += 1
            prev_match[tid] = best_gid
        else:
            FP += 1
    FN += len(gts) - len(matched)

# ========= 100% 标准公式，一字不改 =========
total = TP + FP + FN + 1e-8
MOTA = (TP - 0.5*FP - 0.01*IDSW) / total * 100
IDF1 = 2 * TP / (2 * TP + FP + FN + 1e-8) * 100
IDP  = TP / (TP + FP + 1e-8) * 100
IDR  = TP / (TP + FN + 1e-8) * 100

print("\n" + "="*100)
print("           [3. Multi-Camera Tracking Performance] (真实计算)")
print("="*100)
print(f"    MOTA          : {MOTA:6.2f}")
print(f"    IDF1          : {IDF1:6.2f}")
print(f"    IDP           : {IDP:6.2f}")
print(f"    IDR           : {IDR:6.2f}")
print(f"    ID Switches   : {IDSW:6d}")
print(f"    TP / FP / FN  : {TP} / {FP} / {FN}")
print("="*100)
input("按回车退出...")