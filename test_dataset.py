import torch
from torch.utils.data import DataLoader
from data_loader import RealImageTextAudioDataset, check_paths

# 配置路径（根据您的实际路径修改）
config = {
    "dataset": {
        "image_dir": "D:/multimodel_simulation/data/images/",
        "text_dir": "D:/multimodel_simulation/data/texts/",
        "audio_dir": "D:/multimodel_simulation/data/audios/",
        "label_file": "D:/multimodel_simulation/data/labels.tsv"
    },
    "train": {
        "batch_size": 32
    }
}

def prepare_dataloaders(config):
    # 1. 检查所有文件路径是否有效
    check_paths(
        image_dir=config["dataset"]["image_dir"],
        text_dir=config["dataset"]["text_dir"],
        audio_dir=config["dataset"]["audio_dir"],
        label_file=config["dataset"]["label_file"]
    )

    # 2. 初始化数据集
    train_dataset = RealImageTextAudioDataset(
        image_dir=config["dataset"]["image_dir"],
        text_dir=config["dataset"]["text_dir"],
        audio_dir=config["dataset"]["audio_dir"],
        label_file=config["dataset"]["label_file"],
        tokenizer=None,  # 自动加载BERT tokenizer
        clip_processor=None  # 自动加载CLIP处理器
    )

    # 3. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader

if __name__ == "__main__":
    train_loader = prepare_dataloaders(config)
    
    # 测试第一个batch
    first_batch = next(iter(train_loader))
    print("\n✅ 数据加载成功！第一个batch包含：")
    print(f"图像 tensor形状: {first_batch['image'].shape}")  # [32, 3, 224, 224]
    print(f"文本 token形状: {first_batch['text'].shape}")    # [32, max_seq_len]
    print(f"音频 tensor形状: {first_batch['audio'].shape}")  # [32, audio_max_len]
    print(f"标签 tensor形状: {first_batch['label'].shape}")  # [32]