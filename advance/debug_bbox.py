# check_model.py —— 一键看穿你的 yolo11.pt 长啥样
from ultralytics import YOLO

model = YOLO(r"/models/yolo11/yolo11.pt")

print("你的模型类别名称和顺序如下：")
print(model.names)
print("\n完整打印（带 id）：")
for k, v in model.names.items():
    print(f"class {k} → {v}")