# train_eval.py

import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from multimodal_model import MultimodalModel
from nas_search import draw_architecture


def train_model(config, image_op=None, text_op=None, audio_op=None):
    """
    训练模型使用指定的架构操作。
    如果未提供 image_op/text_op/audio_op，则使用控制器采样。
    """

    from data_loader import RealImageTextAudioDataset

    # 加载数据集
    dataset = RealImageTextAudioDataset(**config['dataset'])

    # 划分训练集和验证集
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [
        int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))
    ])
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

    # 初始化模型
    from multimodal_model import ImageEncoder, TextEncoder, AudioEncoder, MultimodalFusion

    image_encoder = ImageEncoder(num_ops=config['model']['num_ops'])
    text_encoder = TextEncoder(
        input_dim=config['dataset'].get('text_dim', 768),
        hidden_dim=config['model']['text_encoder_hidden_dim']
    )
    audio_encoder = AudioEncoder(
        input_dim=config['dataset'].get('audio_dim', 40),
        hidden_dim=config['model']['audio_encoder_hidden_dim']
    )
    fusion_net = MultimodalFusion(**config['fusion'])
    model = MultimodalModel(image_encoder, text_encoder, audio_encoder, fusion_net).to(config['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    writer = SummaryWriter(log_dir="runs/experiment1")
    best_val_acc = 0

    for epoch in range(config['train']['epochs']):
        model.train()
        correct, total = 0, 0
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")

        for i, batch in enumerate(pbar):
            images = batch['image'].to(config['device'])
            texts = batch['text'].to(config['device'])
            audios = batch['audio'].to(config['device'])
            labels = batch['label'].to(config['device'])

            # 如果未提供架构，则从控制器中采样
            if image_op is None or text_op is None or audio_op is None:
                from nas_search import EnhancedController
                controller = EnhancedController(num_ops=config['model']['num_ops']).to(config['device'])
                controller.load_state_dict(torch.load('checkpoints/nas_controller.pth'))
                image_op_batch, text_op_batch, audio_op_batch = controller.sample(images.size(0), config['device'])
            else:
                image_op_batch = torch.full((images.size(0),), image_op, device=config['device'])
                text_op_batch = torch.full((texts.size(0),), text_op, device=config['device'])
                audio_op_batch = torch.full((audios.size(0),), audio_op, device=config['device'])

            outputs = model(images, texts, audios, image_op_batch, text_op_batch, audio_op_batch)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })

        val_acc = evaluate(config, model, val_loader, image_op, text_op, audio_op)
        writer.add_scalar("Train/Loss", train_loss / len(train_loader), epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        scheduler.step(val_acc)

        print(f"✅ Epoch {epoch+1} | Val Accuracy: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"🏆 Best model saved with accuracy: {best_val_acc:.2f}%")

    writer.close()
    print("🎉 训练完成！")
    return model


def evaluate(config, model, loader, image_op=None, text_op=None, audio_op=None):
    """
    验证模型性能。
    如果未提供架构，则使用控制器采样。
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(config['device'])
            texts = batch['text'].to(config['device'])
            audios = batch['audio'].to(config['device'])
            labels = batch['label'].to(config['device'])

            if image_op is None or text_op is None or audio_op is None:
                from nas_search import EnhancedController
                controller = EnhancedController(num_ops=config['model']['num_ops']).to(config['device'])
                controller.load_state_dict(torch.load('checkpoints/nas_controller.pth'))
                image_op_batch, text_op_batch, audio_op_batch = controller.sample(images.size(0), config['device'])
            else:
                image_op_batch = torch.full((images.size(0),), image_op, device=config['device'])
                text_op_batch = torch.full((texts.size(0),), text_op, device=config['device'])
                audio_op_batch = torch.full((audios.size(0),), audio_op, device=config['device'])

            outputs = model(images, texts, audios, image_op_batch, text_op_batch, audio_op_batch)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


def evaluate_all_architectures(config, controller, val_loader, num_samples=10):
    """
    测试多个采样的架构并返回最优的一个。
    """
    device = config['device']
    controller = controller.to(device)
    results = []

    from multimodal_model import ImageEncoder, TextEncoder, AudioEncoder, MultimodalFusion
    from copy import deepcopy
    import torch

    # 加载验证集
    model_class = MultimodalModel

    with torch.no_grad():
        for idx in range(num_samples):
            image_op, text_op, audio_op = controller.sample(1, device, deterministic=True)
            image_op = image_op.item()
            text_op = text_op.item()
            audio_op = audio_op.item()

            # 构建模型
            model = model_class(
                ImageEncoder(num_ops=config['model']['num_ops']),
                TextEncoder(input_dim=config['dataset'].get('text_dim', 768)),
                AudioEncoder(input_dim=config['dataset'].get('audio_dim', 40)),
                MultimodalFusion(**config['fusion'])
            ).to(device)

            # 加载预训练权重（如果存在）
            try:
                model.load_state_dict(torch.load('checkpoints/best_model.pth'))
            except FileNotFoundError:
                print("⚠️ 未找到最佳模型权重，使用随机初始化评估架构。")

            acc = evaluate(config, model, val_loader, image_op, text_op, audio_op)
            results.append((acc, (image_op, text_op, audio_op)))

            # 可视化该架构
            draw_architecture(
                image_op,
                text_op,
                audio_op,
                filename=f"visualizations/architecture_{idx}"
            )

    # 打印所有架构结果
    results.sort(reverse=True)
    print("\n📊 模型架构评估结果：")
    for acc, ops in results:
        print(f"Image Op: {ops[0]}, Text Op: {ops[1]}, Audio Op: {ops[2]} ➜ Acc: {acc:.2f}%")

    best_acc, best_ops = results[0]
    print(f"\n🏆 最佳架构：{best_ops}, 准确率: {best_acc:.2f}%")
    return best_ops, best_acc


# -------------------------------
# 🧪 主程序入口
# -------------------------------

if __name__ == "__main__":
    import yaml
    import argparse
    from nas_search import EnhancedController

    parser = argparse.ArgumentParser(description="Train or Evaluate a Multimodal Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    args = parser.parse_args()

    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载控制器（用于架构采样）
    controller = EnhancedController(num_ops=config['model']['num_ops']).to(config['device'])
    try:
        controller.load_state_dict(torch.load('checkpoints/nas_controller.pth'))
    except FileNotFoundError:
        print("⚠️ 控制器权重未找到，将使用默认架构进行训练。")

    if args.mode == "train":
        print("🚀 开始训练模型...")
        trained_model = train_model(config)
        print("✅ 模型训练完成。")

    elif args.mode == "evaluate":
        print("🔍 开始评估所有架构...")
        from data_loader import RealImageTextAudioDataset

        dataset = RealImageTextAudioDataset(**config['dataset'])
        _, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
        val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

        best_ops, best_acc = evaluate_all_architectures(config, controller, val_loader, num_samples=10)
        print(f"🎯 最优架构为：Image={best_ops[0]}, Text={best_ops[1]}, Audio={best_ops[2]}, 准确率={best_acc:.2f}%")