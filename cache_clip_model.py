import os
import shutil
from pathlib import Path

import torch
import open_clip
from open_clip import create_model_and_transforms


def cache_clip_model(model_name, pretrained_tag, bin_file_path, commit_id=None):
    """
    将 CLIP 模型权重文件缓存到 huggingface 格式的本地路径中。
    然后手动加载模型权重以避免网络请求。
    """

    # 构建模型标识名（替换非法字符）
    model_identifier = f"timm/{model_name.replace('-', '_')}_clip_224.{pretrained_tag}"

    # 默认 commit id（可以是任意字符串）
    if not commit_id:
        commit_id = "e3f870d679c8970b358d6e6a4c408d2e"  # 固定一个 ID

    # 缓存根目录
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"

    # 构建完整路径
    model_cache_dir = (
        cache_root /
        f"models--{model_identifier.replace('/', '--')}" /
        "snapshots" /
        commit_id
    )

    # 创建目录
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    # 目标路径
    target_path = model_cache_dir / "open_clip_pytorch_model.bin"

    # 复制文件
    if not target_path.exists():
        shutil.copy(bin_file_path, target_path)
        print(f"✅ 模型文件已缓存至: {target_path}")
    else:
        print(f"⚠️ 文件已存在，跳过复制: {target_path}")

    # 手动从本地加载模型
    print("🔄 开始从本地加载模型...")

    model, _, processor = create_model_and_transforms(
        model_name,
        pretrained=None,  # 关键：禁用预训练下载
        precision='fp32',
        device="cpu"
    )

    # 加载本地权重
    checkpoint_path = str(target_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    print("✅ 成功从本地加载 CLIP 模型！")
    return model, processor


if __name__ == "__main__":
    # ======================
    # 🔧 用户配置区
    # ======================

    MODEL_NAME = "ViT-B-32"
    PRETRAINED_TAG = "laion400m_e32"
    BIN_FILE_PATH = r"F:\programs\open_clip_pytorch_model.bin"  # 替换为你自己的路径
    COMMIT_ID = None

    # ======================
    # 🚀 执行缓存 + 加载操作
    # ======================
    model, processor = cache_clip_model(MODEL_NAME, PRETRAINED_TAG, BIN_FILE_PATH, COMMIT_ID)