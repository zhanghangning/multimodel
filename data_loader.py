  # data_loader.py

import os
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchaudio
import numpy as np

from transformers import BertTokenizer
import open_clip
from open_clip import create_model_and_transforms
from pycocotools.coco import COCO


def download_coco_data(save_dir="coco"):
    """
    下载 MS-COCO 数据集（图像 + 标注）。
    如果你已有数据可跳过。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 下载图像
    image_dir = os.path.join(save_dir, "images", "train2017")
    if not os.path.exists(image_dir):
        print("⬇️ 正在下载 MS-COCO 图像...")
        os.makedirs(image_dir, exist_ok=True)
        os.system("curl -L http://images.cocodataset.org/zips/train2017.zip -o train2017.zip")
        os.system("unzip train2017.zip -d " + os.path.dirname(image_dir))
        os.remove("train2017.zip")

    # 下载标注
    ann_dir = os.path.join(save_dir, "annotations")
    if not os.path.exists(ann_dir):
        print("⬇️ 正在下载 MS-COCO 标注...")
        os.makedirs(ann_dir, exist_ok=True)
        os.system("curl -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o ann.zip")
        os.system("unzip ann.zip -d " + save_dir)
        os.remove("ann.zip")

def check_paths(coco_ann_file, image_dir, audio_dir, **kwargs):
    """
    检查数据路径是否存在，若不存在则下载或提示用户
    """
    if not os.path.exists(coco_ann_file):
        raise FileNotFoundError(f"Annotation file not found: {coco_ann_file}. Run download_coco_data() first.")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}. Please check your dataset path.")

    if not os.path.exists(audio_dir):
        print(f"⚠️ Audio directory not found: {audio_dir}, will be created and filled during training.")
        os.makedirs(audio_dir, exist_ok=True)

def download_audio_by_query(query: str, output_dir: str, num_videos=5):
    """
    使用 yt-dlp 根据关键词搜索并下载 YouTube 视频中的音频
    """
    from yt_dlp import YoutubeDL

    output_path = Path(output_dir) / query.replace(" ", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'noplaylist': True,
        'max_downloads': num_videos,
        'quiet': True,
        'default_search': 'ytsearch'
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"{query} sound"])
    print(f"✅ 已下载 {num_videos} 个 '{query}' 类音频到 {output_path}")


class RealImageTextAudioDataset(Dataset):
    """
    使用 MS-COCO 图像+文本 + 自动下载的 YouTube 音频 构建三模态数据集
    用于多模态分类任务
    """

    def __init__(self, 
                 coco_ann_file:str, 
                 image_dir:str, 
                 audio_dir:str,
                 categories=["dog", "cat", "car", "airplane", "person"],
                 max_per_class=100,
                 tokenizer=None,
                 clip_processor=None,
                 max_text_length=50,
                 audio_sample_rate=16000,
                 audio_max_len=16000 * 3):

        self.image_dir = image_dir
        self.audio_dir = audio_dir
        self.max_text_length = max_text_length
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_len = audio_max_len
        self.class_to_idx = {c: i for i, c in enumerate(categories)}

        # 初始化 BERT Tokenizer 和 CLIP Processor
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

        if clip_processor is None:
            # 手动加载本地模型权重
            model, _, self.clip_processor = create_model_and_transforms(
                'ViT-B-32',
                pretrained=None,
                precision='fp32',
                device="cpu"
            )

            # 加载本地权重文件
            local_weights_path = r"C:\Users\Administrator\.cache\huggingface\hub\models--timm--ViT_B_32_clip_224.laion400m_e32\snapshots\e3f870d679c8970b358d6e6a4c408d2e\open_clip_pytorch_model.bin"
            
            state_dict = torch.load(local_weights_path, map_location="cpu")
            model.load_state_dict(state_dict)

            self.clip_model = model  # 可选：保存模型供后续使用
        else:
            self.clip_processor = clip_processor

        # 加载 COCO 数据集 (captions_train2017.json)
        with open(coco_ann_file, 'r') as f:
            ann_data = json.load(f)

        # 构建 image_id 到 captions 的映射
        image_caption_map = {}
        for ann in ann_data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption'].lower()
            if image_id not in image_caption_map:
                image_caption_map[image_id] = []
            image_caption_map[image_id].append(caption)

        # 按照 caption 内容过滤样本
        self.samples = []
        category_counts = {c: 0 for c in categories}
        for image_id, captions in image_caption_map.items():
            matched_categories = [c for c in categories if any(c in cap for cap in captions)]
            if len(matched_categories) == 0:
                continue
            matched_category = matched_categories[0]
            if category_counts[matched_category] >= max_per_class:
                continue
            category_counts[matched_category] += 1
            self.samples.append((image_id, matched_category, random.choice(captions)))

        print(f"✅ 共加载 {len(self.samples)} 个样本，类别：{category_counts}")

    def _load_audio(self, category):
        """
        从本地或自动下载的音频中加载指定类别的音频文件
        """
        audio_path = os.path.join(self.audio_dir, category)
        if not os.path.exists(audio_path):
            download_audio_by_query(category, self.audio_dir, num_videos=5)

        files = list(Path(audio_path).glob("*.wav"))
        if not files:
            return torch.zeros(self.audio_max_len)

        selected = random.choice(files)
        waveform, sr = torchaudio.load(selected, normalize=True)

        if sr != self.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.audio_sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[1] > self.audio_max_len:
            waveform = waveform[:, :self.audio_max_len]
        else:
            pad_len = self.audio_max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform.mean(dim=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, category, caption = self.samples[idx]

        # 加载图像（使用标准 COCO 图像命名规则）
        image_path = os.path.join(self.image_dir, f"{img_id:012d}.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.clip_processor(image).unsqueeze(0)  # [1, C, H, W]

        # 编码文本
        inputs = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        tokenized_text = inputs['input_ids'].squeeze(0)

        # 加载音频
        audio = self._load_audio(category)

        return {
            "image": image,
            "text": tokenized_text,
            "audio": audio,
            "label": self.class_to_idx[category]
        }