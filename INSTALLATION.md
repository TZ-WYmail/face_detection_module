# 安装和使用指南

## ⚡ 最快开始（3 步）

### 1️⃣ 准备视频
```bash
cp your_video.mp4 videos/
```

### 2️⃣ 运行处理
```bash
python face_dedup_pipeline.py videos/your_video.mp4 --cuda
```

### 3️⃣ 查看结果
```bash
ls detected_faces_frontal/
```

**就这么简单！** 模型会在首次运行时**自动下载**。

---

## 📥 模型下载说明

### 自动下载（推荐）
第一次运行脚本时，YOLO 会自动从 GitHub 下载人脸检测模型到 `models/yolo/` 目录。

### 手动下载（网络问题时）

**使用 wget：**
```bash
mkdir -p models/yolo
wget -O models/yolo/yolo26n-face.pt \
  "https://github.com/akanametov/yolo-face/releases/download/v1.0.0/yolo26n-face.pt"
```

**使用 curl：**
```bash
mkdir -p models/yolo
curl -L -o models/yolo/yolo26n-face.pt \
  "https://github.com/akanametov/yolo-face/releases/download/v1.0.0/yolo26n-face.pt"
```

**中国大陆加速（使用镜像）：**
```bash
wget "https://mirror.ghproxy.com/https://github.com/akanametov/yolo-face/releases/download/v1.0.0/yolo26n-face.pt" \
  -O models/yolo/yolo26n-face.pt
```

---

## 🎯 常见用法

### 处理单个视频
```bash
# 使用 GPU（推荐）
python face_dedup_pipeline.py videos/video.mp4 --cuda

# 使用 CPU
python face_dedup_pipeline.py videos/video.mp4
```

### 处理整个目录
```bash
python face_dedup_pipeline.py videos --cuda
```

### 启用人脸跟踪（ByteTrack）
```bash
python face_dedup_pipeline.py videos/video.mp4 --cuda --use-tracks
```

### 高质量提取（严格条件）
```bash
python face_dedup_pipeline.py videos/video.mp4 --cuda \
  --yaw-threshold 10 \
  --pitch-threshold 10 \
  --roll-threshold 5 \
  --confidence 0.7
```

### 快速处理（降采样）
```bash
python face_dedup_pipeline.py videos/video.mp4 --cuda \
  --sample-interval 5
```

---

## 📊 快速参考

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cuda` | 使用 GPU 加速 | CPU 模式 |
| `--use-tracks` | 启用 ByteTrack 跟踪 | 禁用 |
| `--sample-interval N` | 每隔 N 帧处理一次 | 1 |
| `--confidence X` | 检测置信度 (0-1) | 0.5 |
| `--threshold X` | 人脸去重阈值 (0-1) | 0.6 |
| `--yaw-threshold N` | 头部左右角度限制 | 30° |
| `--pitch-threshold N` | 头部上下角度限制 | 30° |
| `--roll-threshold N` | 头部倾斜角度限制 | 30° |
| `--output DIR` | 输出目录 | `detected_faces_frontal/` |

更多参数见：
```bash
python face_dedup_pipeline.py --help
```

---

## ✅ 检查清单

- [ ] Python 3.8+ 已安装
- [ ] 依赖库已安装：`pip install -r requirements.txt`
- [ ] 视频文件已放到 `videos/` 目录
- [ ] （可选）GPU 驱动已安装（用 `--cuda` 时）

---

## 🆘 故障排查

### ❌ 找不到模型

**症状**：运行时报错 "找不到模型"

**解决**：
1. 等待脚本第一次运行时自动下载（可能需要 1-2 分钟）
2. 或手动下载（见上面的"手动下载"章节）

### ❌ 网络连接超时

**症状**：下载时报错 "Connection timeout"

**解决**：
```bash
# 使用镜像源
wget "https://mirror.ghproxy.com/https://github.com/akanametov/yolo-face/releases/download/v1.0.0/yolo26n-face.pt" \
  -O models/yolo/yolo26n-face.pt
```

### ❌ GPU 显存不足

**症状**：运行时报错 "CUDA out of memory"

**解决**：
```bash
# 去掉 --cuda，使用 CPU
python face_dedup_pipeline.py videos/video.mp4

# 或增加采样间隔
python face_dedup_pipeline.py videos/video.mp4 --cuda --sample-interval 5
```

### ❌ 处理很慢

**症状**：原来快，现在变慢

**可能原因与解决**：
- 增加采样间隔：`--sample-interval 5` （每隔 5 帧处理一次）
- 启用 GPU：`--cuda`
- 本地硬盘满，清理空间

---

## 📂 输出文件说明

处理完成后的文件保存位置：

```
detected_faces_frontal/
├── video_name_20240402_001.jpg     # 检测到的正脸图像
├── video_name_20240402_002.jpg
└── ...

output/
├── video-1/
│   ├── face_records_frontal.txt    # 人脸记录（CSV 格式）
│   └── embeddings/
│       ├── face_00001_f0.npy       # 人脸特征向量
│       ├── face_00002_f30.npy
│       └── ...
└── video-2/
    └── ...
```

---

## 💡 工作流示例

```bash
# 1. 放置视频
cp input_video.mp4 videos/

# 2. 处理（启用 GPU 和跟踪）
python face_dedup_pipeline.py videos/input_video.mp4 --cuda --use-tracks

# 3. 查看正脸图像
ls detected_faces_frontal/ | head -10

# 4. 查看人脸特征
ls output/video-1/embeddings/ | head -10

# 5. 读取面部记录（如需进一步处理）
head output/video-1/face_records_frontal.txt
```

---

## 📚 更多文档

- [快速开始指南](QUICKSTART.md) - 详细的使用说明
- [README](README.md) - 项目概览  
- [config.py](config.py) - 所有可配置参数

---

**祝你使用愉快！** 🎉

若有问题，检查日志文件：
```bash
ls -lh logs/
tail -f logs/face_dedup_*.log
```
