# 项目快速开始指南

> **模型自动下载**：运行脚本时，YOLO 会自动下载模型到 `models/yolo/` 目录，无需手动操作。

## 📁 项目结构

```
.
├── videos/                      # 📹 输入视频文件（放你的视频在这里）
├── detected_faces_frontal/      # 📸 输出：检测的正脸图像
├── data/
│   ├── temp/                   # 临时文件
│   └── cache/                  # 缓存文件
├── models/                      # 🤖 模型文件
│   ├── insightface/
│   └── yolo/
├── logs/                        # 📝 运行日志
├── config/                      # ⚙️ 配置文件
├── face_dedup_pipeline.py       # 主程序
├── face_dedup_utils.py          # 工具库
├── config.py                    # 参数配置
└── .env                         # 环境变量
```

## 🚀 快速开始

### 1️⃣ 准备工作

放置你的视频文件到 `videos/` 文件夹：

```bash
cp your_video.mp4 videos/
```

### 2️⃣ 基本用法

**处理单个视频（推荐使用 GPU）：**

```bash
python face_dedup_pipeline.py videos/your_video.mp4 --cuda
```

**处理整个 videos 文件夹：**

```bash
python face_dedup_pipeline.py videos --cuda
```

**仅使用 CPU（不依赖 GPU）：**

```bash
python face_dedup_pipeline.py videos/your_video.mp4
```

### 3️⃣ 查看结果

处理完后，检测的正脸图像会保存到：

```bash
detected_faces_frontal/
├── video_name_001.jpg
├── video_name_002.jpg
└── ...
```

## 📊 常用参数

### 高质量模式（身份证照）

此模式只保留严格正脸，用于需要高质量人脸的场景（如身份证照、签证照片）：

```bash
python face_dedup_pipeline.py videos/your_video.mp4 --cuda \
    --yaw-threshold 10 \      # 左右转动 <10°
    --pitch-threshold 10 \    # 上下抬头 <10°
    --roll-threshold 5 \      # 头部倾斜 <5°
    --confidence 0.7 \        # 检测置信度 >70%
    --threshold 0.6           # 去重相似度 >0.6
```

**参数说明**：
- `--yaw-threshold`: 左右转动角度（0° = 正面，±90° = 侧脸）
- `--pitch-threshold`: 上下抬头角度（0° = 平视，±90° = 低头或抬头）
- `--roll-threshold`: 头部倾斜角度（0° = 水平，±90° = 歪头）

**预期结果**：只保存严格正脸，误检率低，适合身份识别

### 快速处理模式（视频分析）

```bash
python face_dedup_pipeline.py videos/your_video.mp4 --cuda \
    --sample-interval 5 \
    --detector yolo \
    --confidence 0.3
```

### 查看所有参数

```bash
python face_dedup_pipeline.py --help
```

## � 头部姿态过滤详解

### 工作原理

程序通过检测人脸上的5个关键点（两眼、鼻尖、两嘴角），使用 `cv2.solvePnP` 算法计算3D头部姿态角度：

**Yaw（偏航）- 左右转动**
```
正面          左转 30°                 右转 45°
  ↑              ↙                      ↖
 ⭕          ⭙                        ⭚
  |              -30                    +45
```

**Pitch（俯仰）- 上下抬头**
```
抬头 30°         平视                  低头 45°
  ↑            ↑⭕                      ⭕↓
 ⭓             |                       |
 +30           0                      -45
```

**Roll（翻滚）- 头部倾斜**
```
左倾 30°        平衡                  右倾 20°
  ↷⭕           ⭕                      ⭕↷
  +30           0                      -20
```

### 调整策略

**严格模式**（只要身份证等正式证件用的正脸）：
```bash
--yaw-threshold 5 --pitch-threshold 5 --roll-threshold 3
# 只保留极其正面的人脸，误检率最低
```

**正常模式**（推荐）：
```bash
--yaw-threshold 15 --pitch-threshold 15 --roll-threshold 10
# 平衡准度和人脸数量
```

**宽松模式**（接受一些倾偏的人脸）：
```bash
--yaw-threshold 30 --pitch-threshold 25 --roll-threshold 15
# 保留更多人脸变体，但可能包含部分侧脸
```

### 验证过滤效果

运行时启用日志查看每个人脸的姿态检查：

```bash
# 快速测试，只处理10帧看效果
python face_dedup_pipeline.py videos/your_video.mp4 \
    --yaw-threshold 15 \
    --pitch-threshold 15 \
    --roll-threshold 10 \
    --sample-interval 50 \
    -o ./test_pose
```

查看CSV记录中的"头部姿态"列，验证所有人脸的角度都在设定的范围内。

## �📋 日志输出

**去重过程中的日志输出**：

程序会输出每一次去重决策的详细信息：

```
INFO - 🔗 去重匹配: 检测到重复人脸 | 现有ID=00001 | 相似度=0.7324 | 阈值=0.6
INFO - ✨ 新建人脸ID: 00002 | 总数=2 | 头部姿态(Y=-3.5°, P=2.1°, R=1.2°)
```

### 查看调试日志（诊断问题）

如果需要看为什么某些人脸没有被匹配：

```bash
# 设置环境变量启用调试日志
PYTHONWARNINGS=ignore python -u -c "
import logging
logging.getLogger().setLevel(logging.DEBUG)
exec(open('face_dedup_pipeline.py').read())
" videos/your_video.mp4 --cuda
```

或创建一个快速脚本：

```python
# debug_run.py
import logging
import sys

# 启用调试日志
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 运行处理
from face_dedup_pipeline import main
sys.argv = ['face_dedup_pipeline.py', 'videos/your_video.mp4', '--cuda']
main()
```

运行：
```bash
python debug_run.py
```

更多日志详情见 [DEDUP_LOGGING.md](DEDUP_LOGGING.md)

## ⚙️ 配置文件

编辑 `.env` 文件以修改默认配置：

```ini
# 设置是否默认使用 GPU
DEFAULT_CUDA=true

# 采样间隔（每隔 N 帧处理一次，越大处理越快）
SAMPLE_INTERVAL=1

# 检测置信度（0-1，越高越严格）
CONFIDENCE=0.5
```

## 🔧 高级配置

参考 `config.py` 文件了解所有可配置参数：

```bash
cat config.py
```

## 📚 更多帮助

- 完整文档：见 [README.md](README.md)
- 安装指南：见 [SETUP.md](SETUP.md)
- 架构说明：见 [ARCHITECTURE.md](ARCHITECTURE.md)
- 参数详解：见 [config.py](config.py)

## ⚡ 常见问题

### Q: 处理速度太慢？

**Ans:**
1. 启用 GPU：`--cuda`
2. 增加采样间隔：`--sample-interval 5`
3. 切换到轻量检测器：`--detector yolo`

### Q: 显存不足？

**Ans:**
1. 使用 CPU：移除 `--cuda`
2. 增加采样间隔减少内存消耗
3. 处理分辨率更低的视频

### Q: 如何修改输出目录？

**Ans:**
```bash
python face_dedup_pipeline.py videos/your_video.mp4 --output custom_output_dir
```

### Q: 找不到模型？

**Ans:**
首次运行脚本时，YOLO 会自动从 GitHub 下载模型文件（~6 MB）到 `models/yolo/` 目录。

如果仍然缺失，检查：
```bash
ls models/insightface/models/buffalo_l/  # InsightFace 模型
ls models/yolo/                          # YOLO 人脸检测模型
```

若网络问题导致下载失败，可手动下载：
```bash
# 方式 1：使用 wget
mkdir -p models/yolo
wget -O models/yolo/yolo26n-face.pt \
  "https://github.com/akanametov/yolo-face/releases/download/v1.0.0/yolo26n-face.pt"

# 方式 2：使用 curl  
curl -L -o models/yolo/yolo26n-face.pt \
  "https://github.com/akanametov/yolo-face/releases/download/v1.0.0/yolo26n-face.pt"
```

## 💡 工作流示例

```bash
# 1. 放置视频
cp /path/to/video.mp4 videos/

# 2. 处理视频
python face_dedup_pipeline.py videos/video.mp4 --cuda

# 3. 查看结果
ls -lh detected_faces_frontal/

# 4. 查看日志
tail -f logs/*.log  # 如果有日志输出
```

## 📞 获取帮助

- 查看 `--help`：`python face_dedup_pipeline.py --help`
- 查看日志：`logs/` 目录
- 查看完整文档：[README.md](README.md)

---

**现在就开始：**
```bash
python face_dedup_pipeline.py videos/ --cuda
```

祝你使用愉快！ 🎉
