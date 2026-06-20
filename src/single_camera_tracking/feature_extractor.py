# src/single_camera_tracking/feature_extractor.py
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import cv2
import os
from timm.models.vision_transformer import vit_base_patch16_224  # 这行现在能解析了！


class TransReID(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 timm 提供的官方 ViT-Base
        self.backbone = vit_base_patch16_224(pretrained=False, img_size=(256, 128))
        self.backbone.head = nn.Identity()  # 去掉分类头

        # 一个轻量级 neck，输出 768 维特征（和大多数 ReID 模型保持一致）
        self.neck = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 768)
        )

    def forward(self, x):
        x = self.backbone(x)  # [B, 768]
        x = self.neck(x)  # [B, 768]
        return x


class FeatureExtractor:
    def __init__(self, model_path=None):
        if model_path is None:
            from src.utils.config_loader import load_config
            cfg = load_config('configs/model_config.yaml')
            model_path = cfg.get('reid', {}).get('weights_path', 'models/reid/jx_vit_base_p16_224-80ecf9dd.pth')

        if not os.path.exists(model_path):
            model_path = os.path.join('E:/CX_26/CrossCameraMOT/models/reid', os.path.basename(model_path))

        print(f"正在加载 ReID 权重: {model_path}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransReID().to(self.device)

        # 正确加载权重（关键！）
        state_dict = torch.load(model_path, map_location=self.device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # 去除可能不匹配的 head 权重
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print("ReID 模型加载成功！")

        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, frame, detections):
        feats = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                feats.append(torch.zeros(768, device=self.device))
                continue
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = Image.fromarray(crop)
            crop = self.transform(crop).unsqueeze(0).to(self.device)
            feat = self.model(crop)
            feat = feat / feat.norm(p=2, dim=1, keepdim=True)
            feats.append(feat.squeeze(0))
        return feats