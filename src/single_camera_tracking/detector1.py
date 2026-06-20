# src/single_camera_tracking/detector.py   （完整修复版，直接覆盖）

from ultralytics import YOLO
import os
from src.utils.config_loader import load_config


class Detector:
    def __init__(self):
        # 1. 加载配置
        config = load_config('configs/model_config.yaml')

        # 2. 读取 YOLO 权重路径（防止 KeyError）
        yolo_weights = r"E:\CX_26\CrossCameraMOT\models\yolo11\yolo11l.pt"  # 直接写死你的路径

        # 3. 自动补全绝对路径（防止相对路径找不到）
        if not os.path.exists(yolo_weights):
            possible_paths = [
                yolo_weights,
                os.path.join('models', 'yolo11', 'yolo11.pt'),
                os.path.join('E:/CX_26/CrossCameraMOT', 'models/yolo11/yolo11.pt'),
                os.path.join(os.getcwd(), 'models/yolo11/yolo11.pt')
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    yolo_weights = p
                    break
            else:
                raise FileNotFoundError(f"YOLO权重未找到！请检查以下路径是否存在：\n{chr(10).join(possible_paths)}")

        print(f"加载YOLO模型: {yolo_weights}")
        self.model = YOLO(yolo_weights)

    def detect(self, frame):
        """对单帧进行检测，返回结构化结果"""
        results = self.model(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        detections = []
        for box, score, cls in zip(boxes, scores, classes):
            if score < 0.3:  # 过滤低置信度
                continue
            detections.append({
                'bbox': box.tolist(),  # [x1, y1, x2, y2]
                'conf': float(score),
                'class_id': int(cls),
                'class_name': 'special' if cls == 1 else 'ordinary'
            })
        return detections