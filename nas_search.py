
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from collections import deque
import yaml
import argparse
import os
from graphviz import Digraph

# 导入自定义模块
from data_loader import RealImageTextAudioDataset


# -------------------------------
# 📦 增强版 NAS 控制器（支持图像、文本、音频）
# -------------------------------

class EnhancedController(nn.Module):
    def __init__(self, num_ops, hidden_size=64, nhead=4):
        super().__init__()
        self.num_ops = num_ops
        self.hidden_size = hidden_size

        # Embedding 层：包含 None 操作（空操作）
        self.op_embedding = nn.Embedding(num_ops + 1, hidden_size)  # op_idx: 0~num_ops (None is num_ops)

        # LSTM 编码器
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Self Attention 层
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=nhead)

        # 输出头
        self.output_layer = nn.Linear(hidden_size, num_ops)

    def forward(self, prev_actions):
        """
        Args:
            prev_actions: [B, T] 上一个动作序列
        Returns:
            logits: [B, num_ops]
        """
        embedded = self.op_embedding(prev_actions)         # [B, T, D]
        lstm_out, _ = self.lstm(embedded)                 # [B, T, D]
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)  # [B, T, D]
        q_values = self.output_layer(attn_out[:, -1])      # [B, num_ops]
        return q_values

    def sample(self, batch_size, device, deterministic=False):
        """
        从控制器采样架构
        参数:
            batch_size: 采样数量
            device: 设备
            deterministic: 是否确定性采样 (False时使用ε-greedy)
        返回:
            image_op: 图像操作选择 [batch_size]
            text_op: 文本操作选择 [batch_size]
            audio_op: 音频操作选择 [batch_size]
        """
        with torch.no_grad():
            state = torch.full((batch_size, 1),
                               fill_value=self.num_ops,
                               dtype=torch.long,
                               device=device)
            image_op = self.act(state, epsilon=0 if deterministic else 0.1)
            text_op = self.act(state, epsilon=0 if deterministic else 0.1)
            audio_op = self.act(state, epsilon=0 if deterministic else 0.1)
            return image_op, text_op, audio_op

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return torch.randint(0, self.num_ops, (state.size(0),)).to(state.device)
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax(dim=-1)


# -------------------------------
# 🧠 经验回放缓冲区（用于强化学习）
# -------------------------------

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, states, actions, rewards, next_states):
        """存储批量经验"""
        for i in range(states.size(0)):
            self.buffer.append((
                states[i].clone().detach().cpu(),
                int(actions[i].item()),
                float(rewards[i].item()),  # 确保是 Python float
                next_states[i].clone().detach().cpu()
            ))

    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states)
        )

    def __len__(self):
        return len(self.buffer)


# -------------------------------
# 🚀 NAS 训练主函数
# -------------------------------

def train_nas(config):
    from data_loader import RealImageTextAudioDataset
    from multimodal_model import ImageEncoder, TextEncoder, AudioEncoder, MultimodalFusion, MultimodalModel

    dataset_config = config['dataset']

    # ✅ 添加路径检查
    from data_loader import check_paths
    check_paths(**dataset_config)

    # 加载真实三模态数据集
    dataset = RealImageTextAudioDataset(**dataset_config)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [
        int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))
    ])
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['train']['batch_size'])

    # 初始化模型
    image_encoder = ImageEncoder(num_ops=config['model']['num_ops'])
    text_encoder = TextEncoder(input_dim=config['dataset'].get('text_dim', 768),
                               hidden_dim=config['model']['text_encoder_hidden_dim'])
    audio_encoder = AudioEncoder(input_dim=config['dataset'].get('audio_dim', 40),
                                 hidden_dim=config['model']['audio_encoder_hidden_dim'])
    fusion_net = MultimodalFusion(**config['fusion'])
    model = MultimodalModel(image_encoder, text_encoder, audio_encoder, fusion_net).to(config['device'])

    # 初始化控制器
    controller = EnhancedController(num_ops=config['model']['num_ops']).to(config['device'])
    target_controller = EnhancedController(num_ops=config['model']['num_ops']).to(config['device'])
    target_controller.load_state_dict(controller.state_dict())
    target_controller.eval()

    # 优化器
    # 强制转换为 float 防止 string 类型导致错误
    try:
        config['nas']['controller_lr'] = float(config['nas']['controller_lr'])
    except ValueError:
        raise ValueError("❌ config['nas']['controller_lr'] must be a number (e.g. 3e-4), got:", config['nas']['controller_lr'])
    controller_optim = optim.Adam(controller.parameters(), lr=config['nas']['controller_lr'])
    try:
        config['nas']['model_lr'] = float(config['nas']['model_lr'])
    except ValueError:
        raise ValueError("❌ config['nas']['model_lr'] must be a number (e.g. 1e-3), got:", config['nas']['model_lr'])
    model_optim = optim.Adam(model.parameters(), lr=config['nas']['model_lr'])

    criterion = nn.CrossEntropyLoss()
    replay_buffer = ReplayBuffer(capacity=10000)

    gamma = config['nas'].get('gamma', 0.99)
    epsilon = config['nas'].get('epsilon', 1.0)
    min_epsilon = config['nas'].get('min_epsilon', 0.05)
    update_target_steps = config['nas'].get('update_target_steps', 100)
    total_steps = 0

    for epoch in range(config['nas']['epochs']):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['nas']['epochs']}")
        for i, batch in enumerate(pbar):
            images = batch['image'].to(config['device'])
            texts = batch['text'].to(config['device'])
            audios = batch['audio'].to(config['device'])
            labels = batch['label'].to(config['device'])

            # Step 1: ε-greedy 采样操作
            with torch.no_grad():
                prev_action = torch.full((images.size(0), 1),
                                        fill_value=config['model']['num_ops'],
                                        dtype=torch.long,
                                        device=config['device'])
                image_op = controller.act(prev_action, epsilon=epsilon)
                text_op = controller.act(prev_action, epsilon=epsilon)
                audio_op = controller.act(prev_action, epsilon=epsilon)

            # Step 2: 模型前向传播
            outputs = model(images, texts, audios, image_op, text_op, audio_op)
            loss = criterion(outputs, labels)

            # Step 3: 更新模型参数
            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

            # Step 4: 计算奖励（负交叉熵）
            with torch.no_grad():
                reward = -F.cross_entropy(outputs, labels).item()
                reward_tensor = torch.tensor([reward] * images.size(0), device=config['device'])

            # Step 5: 构造下一个状态并存入 buffer
            next_state = torch.cat([prev_action, image_op.unsqueeze(-1)], dim=1)
            replay_buffer.push(prev_action, image_op, reward_tensor, next_state)

            # Step 6: 如果 buffer 中样本足够，开始训练控制器
            if len(replay_buffer) > config['train']['batch_size']:
                states, actions, rewards, next_states = replay_buffer.sample(config['train']['batch_size'])
                states = states.to(config['device'])
                actions = actions.to(config['device'])
                rewards = rewards.to(config['device'])
                next_states = next_states.to(config['device'])

                current_q_values = controller(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

                with torch.no_grad():
                    next_q_values = target_controller(next_states).max(1)[0]
                    expected_q_values = rewards + gamma * next_q_values

                loss_ctl = F.mse_loss(current_q_values, expected_q_values)

                controller_optim.zero_grad()
                loss_ctl.backward()
                controller_optim.step()

            # Step 7: 更新目标网络
            if total_steps % update_target_steps == 0:
                target_controller.load_state_dict(controller.state_dict())

            # Step 8: 衰减 epsilon
            epsilon = max(min_epsilon, epsilon * config['nas']['epsilon_decay'])
            total_steps += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })

    # 保存控制器权重
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(controller.state_dict(), 'checkpoints/nas_controller.pth')
    print("✅ NAS Controller trained and saved.")
    return controller


# -------------------------------
# 🎨 可视化控制器选择的架构图
# -------------------------------

def draw_architecture(image_op, text_op, audio_op, filename="sampled_architecture",
                     image_ops=None, text_ops=None, audio_ops=None):
    """
    使用 graphviz 绘制多模态架构图。
    """
    if image_ops is None:
        image_ops = {
            0: "CNN",
            1: "ResNet",
            2: "Vision Transformer"
        }
    if text_ops is None:
        text_ops = {
            0: "LSTM",
            1: "GRU",
            2: "Transformer"
        }
    if audio_ops is None:
        audio_ops = {
            0: "MFCC",
            1: "Wav2Vec2",
            2: "AST"
        }

    dot = Digraph(comment='Multimodal Architecture', format='png')
    dot.attr(rankdir='LR')

    with dot.subgraph(name='cluster_image') as img_sub:
        img_sub.attr(label='Image Encoder', color='blue')
        img_sub.node('input_image', 'Input Image')
        img_sub.node(f'img_op_{image_op}', f'{image_ops[image_op]}')
        img_sub.edge('input_image', f'img_op_{image_op}')

    with dot.subgraph(name='cluster_text') as txt_sub:
        txt_sub.attr(label='Text Encoder', color='green')
        txt_sub.node('input_text', 'Input Text')
        txt_sub.node(f'txt_op_{text_op}', f'{text_ops[text_op]}')
        txt_sub.edge('input_text', f'txt_op_{text_op}')

    with dot.subgraph(name='cluster_audio') as aud_sub:
        aud_sub.attr(label='Audio Encoder', color='red')
        aud_sub.node('input_audio', 'Input Audio')
        aud_sub.node(f'aud_op_{audio_op}', f'{audio_ops[audio_op]}')
        aud_sub.edge('input_audio', f'aud_op_{audio_op}')

    with dot.subgraph(name='cluster_fusion') as fus_sub:
        fus_sub.attr(label='Fusion & Classifier', color='orange')
        fus_sub.node('fusion', 'Multimodal Fusion')
        fus_sub.node('classifier', 'Classifier')

    # 模块连接
    dot.edge(f'img_op_{image_op}', 'fusion')
    dot.edge(f'txt_op_{text_op}', 'fusion')
    dot.edge(f'aud_op_{audio_op}', 'fusion')
    dot.edge('fusion', 'classifier')

    dot.render(filename, view=False)
    print(f"✅ 架构图已保存为 {filename}.png")


# -------------------------------
# 🧪 主程序入口（可选：训练完后自动绘制最佳架构）
# -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NAS search")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    # ✅ 使用 utf-8 显式读取配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    controller = train_nas(config)

    # 采样最佳架构并可视化
    device = config['device']
    controller = controller.to(device)

    with torch.no_grad():
        image_op, text_op, audio_op = controller.sample(1, device)

    print(f"Sampled architecture - Image op: {image_op.item()}, Text op: {text_op.item()}, Audio op: {audio_op.item()}")

    draw_architecture(image_op.item(), text_op.item(), audio_op.item(), "visualizations/best_architecture")