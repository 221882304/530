# src/utils/config_loader.py (修改后)
import yaml
import os

def load_config(file_path):
    full_path = os.path.join(os.path.dirname(__file__), '..', '..', file_path)  # 相对路径调整，如果需要
    with open(file_path, 'r', encoding='utf-8') as f:  # 指定 UTF-8 编码
        return yaml.safe_load(f)