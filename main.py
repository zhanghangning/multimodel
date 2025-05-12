# -*- coding: utf-8 -*-
# main.py

import yaml
import torch
import argparse
from train_eval import train_model, evaluate
# main.py

import yaml
import torch
import argparse
from train_eval import train_model, evaluate
from nas_search import train_nas
from multimodal_model import MultimodalModel, ImageEncoder, TextEncoder, AudioEncoder, MultimodalFusion
from data_loader import RealImageTextAudioDataset
from torch.utils.data import DataLoader

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("🔧 Loaded Config:\n", config)  # 打印整个 config 查看是否还有其他类型错误
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return config

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Run Multi-Modal NAS and Training Pipeline")

    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")

    parser.add_argument("--mode", choices=["nas", "train", "all"], default="all",
                        help="Mode: 'nas' for architecture search, 'train' for retraining, 'all' for both")

    return parser.parse_args()


def run_nas(config):
    """
    运行 NAS 搜索流程
    """
    print("🚀 开始进行 NAS 架构搜索...")

    # 加载真实数据集
    dataset = RealImageTextAudioDataset(
        coco_ann_file=config['dataset']['coco_ann_file'],
        image_dir=config['dataset']['image_dir'],
        audio_dir=config['dataset']['audio_dir'],
        categories=config['dataset'].get('categories', ["dog", "cat", "car", "airplane", "person"]),
        max_per_class=config['dataset'].get('max_per_class', 100)
    )
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [
        int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))
    ])
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

    # 训练 NAS 控制器
    controller = train_nas(config)

    # 可视化最佳架构
    visualize_best_architecture(config)

    # 返回控制器以供后续训练使用
    return controller


def visualize_best_architecture(config):
    """
    可视化 NAS 搜索到的最佳架构
    """
    from nas_search import EnhancedController

    device = config['device']
    controller = EnhancedController(num_ops=config['model']['num_ops'])
    controller.load_state_dict(torch.load('checkpoints/nas_controller.pth'))
    controller = controller.to(device)

    with torch.no_grad():
        best_image_op, best_text_op, best_audio_op = controller.sample(1, device)
    best_image_op = best_image_op.item()
    best_text_op = best_text_op.item()
    best_audio_op = best_audio_op.item()

    print(f"\n📊 最佳架构可视化：\n"
          f"🖼️ 图像编码操作 ID: {best_image_op}\n"
          f"📝 文本编码操作 ID: {best_text_op}\n"
          f"🔊 音频编码操作 ID: {best_audio_op}")


def run_training_with_best_architecture(config, controller=None):
    """
    使用 NAS 搜索到的最佳架构进行模型训练
    """
    print("🚀 使用最佳架构开始训练多模态模型...")

    device = config['device']

    # 如果未提供控制器，则加载已保存的 NAS 控制器权重
    if controller is None:
        from nas_search import EnhancedController
        controller = EnhancedController(num_ops=config['model']['num_ops'])
        controller.load_state_dict(torch.load('checkpoints/nas_controller.pth'))
        controller = controller.to(device)

    # 获取最佳架构
    with torch.no_grad():
        best_image_op, best_text_op, best_audio_op = controller.sample(1, device)
    best_image_op = best_image_op.item()
    best_text_op = best_text_op.item()
    best_audio_op = best_audio_op.item()

    print(f"🏆 最佳架构："
          f"Image Op={best_image_op}, "
          f"Text Op={best_text_op}, "
          f"Audio Op={best_audio_op}")

    # 初始化并训练模型
    model = train_model(config, best_image_op, best_text_op, best_audio_op)

    # 测试模型性能
    test_dataset = RealImageTextAudioDataset(**config['test_dataset'])
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'])

    best_model = MultimodalModel(
        ImageEncoder(num_ops=config['model']['num_ops']),
        TextEncoder(input_dim=config['dataset'].get('text_dim', 768)),
        AudioEncoder(input_dim=config['dataset'].get('audio_dim', 40)),
        MultimodalFusion(**config['fusion'])
    ).to(device)

    best_model.load_state_dict(torch.load('checkpoints/best_model.pth'))

    test_acc = evaluate(config, best_model, test_loader, best_image_op, best_text_op, best_audio_op)
    print(f"\n🏁 最终测试准确率: {test_acc:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    device = config['device']
    print(f"Using device: {device}")

    controller = None

    if args.mode in ["nas", "all"]:
        controller = run_nas(config)

    if args.mode in ["train", "all"]:
        run_training_with_best_architecture(config, controller)