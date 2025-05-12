import os
from pathlib import Path

# 配置路径（根据实际情况修改）
data_dir = Path("D:/multimodel_simulation/data/")
image_dir = data_dir / "images"
text_dir = data_dir / "texts"
audio_dir = data_dir / "audios"
output_file = data_dir / "labels.tsv"

# 获取文件名（不带扩展名）
image_files = {f.stem: f.name for f in image_dir.glob("*") if f.suffix in [".jpg", ".png"]}
text_files = {f.stem: f.name for f in text_dir.glob("*.txt")}
audio_files = {f.stem: f.name for f in audio_dir.glob("*.wav")}

# 检查一致性
common_keys = set(image_files) & set(text_files) & set(audio_files)
if len(common_keys) == 0:
    raise ValueError("❌ 错误：未找到匹配的图像、文本、音频文件")

# 生成标签文件
with open(output_file, "w", encoding="utf-8") as f:
    for idx, key in enumerate(common_keys):
        # 此处用索引作为示例标签，实际应替换为真实标签
        line = f"{image_files[key]}\t{text_files[key]}\t{audio_files[key]}\t{idx % 10}\n"
        f.write(line)

print(f"✅ 已生成标签文件：{output_file}")
print(f"生成样本数：{len(common_keys)}")