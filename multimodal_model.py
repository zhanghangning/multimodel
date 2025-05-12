# multimodal_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# 🖼️ 图像编码器（支持多种操作）
# -------------------------------

class ImageEncoder(nn.Module):
    def __init__(self, num_ops=4):
        super().__init__()
        self.num_ops = num_ops

        # 模拟不同图像编码器
        self.encoders = nn.ModuleList([
            self._make_cnn(),          # Op 0: CNN
            self._make_resnet(),       # Op 1: ResNet-like
            self._make_vit(),          # Op 2: Vision Transformer
            self._make_identity()      # Op 3: Identity (passthrough)
        ])

        # 输出维度统一为 256
        self.output_proj = nn.Linear(512, 256)

    def _make_cnn(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def _make_resnet(self):
        from torchvision.models import resnet18
        resnet = resnet18(pretrained=False)
        modules = list(resnet.children())[: -2]
        return nn.Sequential(*modules)

    def _make_vit(self):
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Identity()
        return model

    def _make_identity(self):
        return nn.Identity()

    def forward(self, x, op_idx=None):
        """
        Args:
            x: 输入图像 [B, C, H, W]
            op_idx: 编码器选择（0~num_ops-1）
        Returns:
            image_feat: [B, D]
        """
        if op_idx is None:
            op_idx = random.randint(0, self.num_ops - 1)

        if op_idx == 2:  # Vision Transformer 需要调整输入尺寸
            x = F.interpolate(x, size=(224, 224))

        out = self.encoders[op_idx](x)
        if isinstance(out, list) or len(out.shape) > 2:
            out = out.mean(dim=(-1, -2)) if len(out.shape) == 4 else out.mean(dim=1)

        return self.output_proj(out)


# -------------------------------
# 📝 文本编码器（支持 LSTM / GRU / Transformer）
# -------------------------------

class TextEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_ops = 3

        # 模拟不同文本编码器
        self.encoders = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim, batch_first=True),     # Op 0: LSTM
            nn.GRU(input_dim, hidden_dim, batch_first=True),      # Op 1: GRU
            nn.TransformerEncoder(                               # Op 2: Transformer
                nn.TransformerEncoderLayer(d_model=input_dim, nhead=8), num_layers=2)
        ])

        self.output_proj = nn.Linear(hidden_dim if input_dim != hidden_dim else input_dim, 128)

    def forward(self, x, op_idx=None):
        """
        Args:
            x: 输入文本 [B, SeqLen]
            op_idx: 编码器选择（0~2）
        Returns:
            text_feat: [B, D]
        """
        if op_idx is None:
            op_idx = random.randint(0, self.num_ops - 1)

        if op_idx < 2:  # LSTM / GRU
            out, _ = self.encoders[op_idx](x)
            out = out[:, -1]  # 取最后一个时间步
        else:  # Transformer
            out = self.encoders[op_idx](x)

        return self.output_proj(out)


# -------------------------------
# 🔊 音频编码器（支持 MFCC / Wav2Vec2 / AST）
# -------------------------------

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_ops = 3

        # 模拟不同音频编码器
        self.encoders = nn.ModuleList([
            nn.Sequential(                             # Op 0: MFCC-style CNN
                nn.Conv1d(1, 32, kernel_size=5),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            ),
            nn.LSTM(input_dim, hidden_dim, batch_first=True),  # Op 1: LSTM
            nn.TransformerEncoder(                     # Op 2: Transformer
                nn.TransformerEncoderLayer(d_model=input_dim, nhead=4), num_layers=2)
        ])

        self.output_proj = nn.Linear(hidden_dim if input_dim != hidden_dim else input_dim, 128)

    def forward(self, x, op_idx=None):
        """
        Args:
            x: 输入音频 [B, T]
            op_idx: 编码器选择（0~2）
        Returns:
            audio_feat: [B, D]
        """
        if op_idx is None:
            op_idx = random.randint(0, self.num_ops - 1)

        x = x.unsqueeze(1) if op_idx == 0 else x.unsqueeze(1)  # 扩展通道维度

        if op_idx == 0:  # MFCC-style CNN
            out = self.encoders[op_idx](x)
        elif op_idx == 1:  # LSTM
            out, _ = self.encoders[op_idx](x)
            out = out[:, -1]
        else:  # Transformer
            out = self.encoders[op_idx](x)
            out = out.mean(dim=1)

        return self.output_proj(out)


# -------------------------------
# 🔗 多模态融合模块
# -------------------------------

class MultimodalFusion(nn.Module):
    def __init__(self, fusion_type="concat", output_size=128, dropout_rate=0.5):
        super().__init__()
        self.fusion_type = fusion_type
        self.output_size = output_size

        # 假设每个模态输出是 128 维
        input_total = 128 * 3 if fusion_type == "concat" else 128

        if fusion_type == "concat":
            self.fuser = nn.Sequential(
                nn.Linear(input_total, output_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(output_size)
            )
        elif fusion_type == "attention":
            self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4)
            self.fuser = nn.Sequential(
                nn.Linear(128, output_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        elif fusion_type == "gated":
            self.gate = nn.Linear(128 * 3, 3)
            self.fuser = nn.Sequential(
                nn.Linear(128 * 3, output_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

    def forward(self, img_feat, txt_feat, aud_feat):
        """
        Args:
            img_feat: [B, D]
            txt_feat: [B, D]
            aud_feat: [B, D]
        Returns:
            fused_feat: [B, output_size]
        """
        if self.fusion_type == "concat":
            combined = torch.cat([img_feat, txt_feat, aud_feat], dim=-1)
            return self.fuser(combined)
        elif self.fusion_type == "attention":
            feats = torch.stack([img_feat, txt_feat, aud_feat], dim=0)
            out, _ = self.attn(feats, feats, feats)
            return self.fuser(out.mean(dim=0))
        elif self.fusion_type == "gated":
            combined = torch.cat([img_feat, txt_feat, aud_feat], dim=-1)
            gate_weights = F.softmax(self.gate(combined), dim=-1)
            weighted = img_feat * gate_weights[:, 0:1] + \
                       txt_feat * gate_weights[:, 1:2] + \
                       aud_feat * gate_weights[:, 2:3]
            return self.fuser(weighted)
        else:
            raise NotImplementedError


# -------------------------------
# 🧠 完整的多模态模型
# -------------------------------

class MultimodalModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, audio_encoder, fusion_net):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.fusion = fusion_net
        self.classifier = nn.Linear(fusion_net.output_size, 5)  # 5个类别：dog/cat/car/airplane/person

    def forward(self, images, texts, audios, image_op=None, text_op=None, audio_op=None):
        """
        Args:
            images: [B, C, H, W]
            texts: [B, SeqLen]
            audios: [B, T]
            image_op: 图像编码器选择
            text_op: 文本编码器选择
            audio_op: 音频编码器选择
        Returns:
            logits: [B, NumClasses]
        """
        img_feat = self.image_encoder(images, image_op)
        txt_feat = self.text_encoder(texts, text_op)
        aud_feat = self.audio_encoder(audios, audio_op)

        fused = self.fusion(img_feat, txt_feat, aud_feat)
        return self.classifier(fused)