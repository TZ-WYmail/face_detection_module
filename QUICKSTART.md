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

## � 初始化与环境检查

> **首次使用必读**：在运行任何处理脚本前，必须完成项目初始化。此步骤验证依赖、模型和环境配置。

### 前置步骤：创建 Conda 环境（可选但推荐）

> 使用 Conda 虚拟环境可以隔离项目依赖，避免版本冲突。此步骤**可选但强烈推荐**。

#### 方式 A：快速方式（一行命令）

```bash
# 创建 Python 3.10 的 conda 环境
conda create -n face_detect python=3.10 -y

# 激活环境
conda activate face_detect

# 安装依赖
pip install -r requirements.txt

# 运行初始化
python initialize_project.py
```

#### 方式 B：详细步骤

**1️⃣ 查看可用 Python 版本**
```bash
conda search python | grep -E "^python\s+(3\.(10|11|12))"
```

预期输出：
```
python                        3.10.0             h62ddb7d_0  conda-forge
python                        3.11.0             h62ddb7d_0  conda-forge
python                        3.12.0             h62ddb7d_0  conda-forge
```

**2️⃣ 创建指定 Python 版本的环境**

```bash
# 方式1：指定具体小版本（推荐 Python 3.10）
conda create -n face_detect python=3.10 -y

# 方式2：指定版本范围（Python 3.10.x 的最新版）
conda create -n face_detect "python>=3.10,<3.11" -y

# 方式3：使用 YAML 配置文件（见下文）
conda env create -f environment.yml
```

**3️⃣ 激活环境**
```bash
# Linux / macOS
conda activate face_detect

# Windows
conda activate face_detect
```

验证激活成功（终端左侧应显示 `(face_detect)`）：
```bash
python --version  # 应该显示 Python 3.10.x
which python      # 应该显示 conda 环境路径
```

**4️⃣ 安装项目依赖**
```bash
pip install -r requirements.txt
```

#### 方式 C：使用 environment.yml（完全复现环境）

项目已提供 `environment.yml` 文件，包含所有锁定的依赖版本：

```bash
# 创建环境（自动使用 environment.yml）
conda env create -f environment.yml

# 激活环境
conda activate face_detect

# 验证
python initialize_project.py
```

#### Conda 环境管理常用命令

```bash
# 查看所有环境
conda env list

# 激活环境
conda activate face_detect

# 退出环境（回到 base）
conda deactivate

# 删除环境
conda remove -n face_detect --all -y

# 克隆环境
conda create --clone face_detect -n face_detect_backup

# 导出环境配置
conda env export > my_environment.yml

# 更新环境中的包
conda update -n face_detect --all

# 在环境中运行命令（无需激活）
conda run -n face_detect python face_dedup_pipeline.py videos/video.mp4 --cuda
```

#### Conda 环境推荐配置

| 环境特性 | Python 版本 | PyTorch 版本 | 用途 |
|---------|---------|----------|------|
| **标准环境** | 3.10 | 2.0+ | 推荐，兼容性好 |
| **最新环境** | 3.11 或 3.12 | 2.3+ | 最佳性能，部分包可能不兼容 |
| **保守环境** | 3.9 | 1.13 | 旧设备/旧包依赖 |
| **GPU 优化** | 3.10 + CUDA 12.1 | 2.0+ (cuda) | GPU 加速处理 |

---

### 初始化步骤

#### 第1步：运行初始化检查
```bash
python initialize_project.py
```

此脚本自动检查：
- ✅ **Python依赖**：OpenCV、NumPy、PyTorch、UltraYOLOS、ONNX Runtime 等
- ✅ **模型完整性**：InsightFace（w600k_r50.onnx 等）、YOLO 人脸检测模型
- ✅ **项目结构**：必需的配置文件、处理脚本是否完整
- ✅ **写入权限**：models/、output/、logs/ 等目录的写入权限
- ✅ **参数系统**：命令行参数解析是否正常

#### 第2步：根据检查结果处理

**✅ 所有检查通过**
```
✓ 所有检查已通过！项目已准备就绪
```
→ 直接跳到第 3️⃣ 步运行处理

**⚠️ 缺失 Python 依赖**
```
✗ opencv-python 未安装
✗ torch 未安装
```
→ 运行安装命令：
```bash
pip install -r requirements.txt
```

**⚠️ 缺失模型文件**
```
✗ InsightFace 模型不完整
✗ YOLO 人脸检测模型缺失
```
→ 下载模型：
```bash
# 下载 InsightFace 特征提取模型
python download_insightface.py

# 下载 YOLO 人脸检测模型（自动）
python download_models.py
```

#### 第3步：验证初始化完成

再次运行初始化脚本，确认所有检查都通过：
```bash
python initialize_project.py
```

期望输出：
```
✓ opencv-python
✓ numpy
✓ torch
✓ ultralytics
✓ onnxruntime

......[模型检查]......
✓ InsightFace 特征提取模型存在
✓ YOLO 人脸检测模型存在

......[项目结构检查]......
✓ config.py
✓ face_dedup_pipeline.py
✓ face_dedup_utils.py

✅ 所有检查已通过！项目已准备就绪
```

---

### 初始化常见问题

#### Q1：为什么要初始化？
**A:** 初始化确保所有依赖和模型都已安装。缺失任何一个都会导致处理失败。节省调试时间，直接定位问题。

#### Q2：初始化失败怎么办？
**A:** 逐一检查失败项：
- **依赖缺失** → 运行 `pip install -r requirements.txt`
- **模型缺失** → 运行 `python download_insightface.py && python download_models.py`
- **权限问题** → 检查 models/、output/、logs/ 目录权限：`chmod 755 models output logs`
- **网络问题** → 检查网络连接（下载模型需要网络）

#### Q3：需要多长时间？
**A:** 
- 首次初始化（包括下载模型）：**5-15 分钟**（依赖网速）
- 后续初始化（仅检查）：**10-20 秒**

#### Q4：模型下载失败？
**A:** 常见原因及解决方案：

```bash
# 情况1：网络超时
→ 重试下载
python download_insightface.py
python download_models.py

# 情况2：空间不足
→ 检查磁盘空间（需要 ~500 MB）
df -h

# 情况3：权限不足
→ 创建模型目录
mkdir -p models/insightface models/yolo
chmod 777 models

# 情况4：代理/防火墙
→ 配置代理或使用 VPN
```

#### Q5：关于 Conda 环境？
**A:** 常见 Conda 问题与解决方案：

**问题 1：找不到 conda 命令**
```bash
# 检查 Conda 是否安装
conda --version

# 若未安装，下载 Miniconda
# macOS/Linux: https://docs.conda.io/projects/miniconda/en/latest/
# Windows: https://docs.conda.io/projects/miniconda/en/latest/miniconda-latest-Windows-x86_64.exe

# 安装后重启终端或运行
source ~/.bashrc  # Linux/macOS
```

**问题 2：激活环境失败**
```bash
# 查看环境是否存在
conda env list

# 若不存在，创建环境
conda create -n face_detect python=3.10 -y

# 激活环境
conda activate face_detect
```

**问题 3：Python 版本不对**
```bash
# 检查当前 Python 版本
python --version

# 若不是 3.10+，删除环境重建
conda remove -n face_detect --all -y
conda create -n face_detect python=3.10 -y
conda activate face_detect
```

**问题 4：Conda 环境中的包依然缺失**
```bash
# 使用 --no-deps 强制安装
pip install --upgrade --force-reinstall -r requirements.txt

# 或清除 pip 缓存后重装
pip cache purge
pip install -r requirements.txt
```

**问题 5：在虚拟环境中运行脚本**
```bash
# 方式1：激活后运行（推荐）
conda activate face_detect
python face_dedup_pipeline.py videos/video.mp4 --cuda

# 方式2：无需激活，直接运行
conda run -n face_detect python face_dedup_pipeline.py videos/video.mp4 --cuda

# 方式3：使用完整路径
~/miniconda3/envs/face_detect/bin/python face_dedup_pipeline.py videos/video.mp4 --cuda
```

---

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

## ⚙️ 配置文件详解

### .env 环境变量配置

编辑项目根目录的 `.env` 文件以修改默认配置：

```ini
# ==================== GPU 配置 ====================
# 是否默认使用 GPU（true/false）
DEFAULT_CUDA=true

# GPU 设备索引（多GPU情况下指定，默认0）
GPU_DEVICE=0

# ==================== 处理参数 ====================
# 采样间隔（每隔 N 帧处理一次，越大处理越快）
# 推荐：
#   - 视频分析：5-10（快速处理）
#   - 一般处理：2-3（平衡速度和质量）
#   - 高质量模式：1（逐帧处理）
SAMPLE_INTERVAL=1

# 检测置信度阈值（0-1，越高越严格，越低检测越多）
# 推荐：
#   - 严格模式：0.7-0.8（高质量人脸）
#   - 正常模式：0.5-0.6（推荐）
#   - 宽松模式：0.3-0.4（最大化人脸数量）
CONFIDENCE=0.5

# 人脸去重相似度阈值（0-1，越高越严格）
# 推荐：
#   - 严格模式：0.9（几乎无去重）
#   - 正常模式：0.8（推荐，平衡）
#   - 宽松模式：0.6-0.7（积极去重）
DEDUP_THRESHOLD=0.8

# ==================== 模型选择 ====================
# 人脸检测器（insightface / yolo）
# insightface：精确度高，速度较慢
# yolo：速度快，精确度次之
DETECTOR=yolo

# 识别/特征提取模型（insightface/w600k_r50.onnx）
RECOGNITION_MODEL=insightface

# ==================== 输出配置 ====================
# 检测到的正脸图像保存目录
OUTPUT_DIR=./detected_faces_frontal

# 输出视频保存目录（可选）
VIDEO_OUTPUT_DIR=./videos_with_boxes

# 日志输出级别（DEBUG / INFO / WARNING / ERROR）
LOG_LEVEL=INFO

# ==================== 性能优化 ====================
# 最大批处理大小（GPU内存允许的情况下）
BATCH_SIZE=32

# 是否启用多进程处理
ENABLE_MULTIPROCESSING=false

# 工作进程数（仅在多进程启用时有效）
NUM_WORKERS=4
```

### config.py 配置参数

项目核心配置在 `config.py` 中存储。主要参数包括：

```python
# 人脸检测和识别
DETECTOR = "yolo"                     # 检测器类型
YOLO_CONFIDENCE = 0.5                 # YOLO 置信度
YOLO_IOU_THRESHOLD = 0.5              # YOLO IOU 阈值

# 人脸去重
DEDUP_THRESHOLD = 0.8                 # 去重相似度阈值
STRICT_DEDUP_THRESHOLD = 0.9          # 严格去重阈值

# 头部姿态过滤
YAW_THRESHOLD = 15                    # 左右转动阈值（度）
PITCH_THRESHOLD = 15                  # 上下抬头阈值（度）
ROLL_THRESHOLD = 10                   # 头部倾斜阈值（度）

# 采样和处理
DEFAULT_SAMPLE_INTERVAL = 1           # 默认采样间隔
BATCH_SIZE = 32                       # 批处理大小
```

### 修改配置的三种方式

#### 方式 1️⃣：命令行参数（推荐用于一次性改动）
```bash
# 覆盖默认配置
python face_dedup_pipeline.py videos/video.mp4 \
    --cuda \
    --threshold 0.9 \
    --sample-interval 5 \
    --confidence 0.7 \
    --yaw-threshold 10
```

**优点**：灵活，适合实验；**缺点**：每次都要输入

#### 方式 2️⃣：修改 .env 文件（推荐用于持久配置）
```bash
# 编辑 .env
nano .env

# 然后运行脚本（自动读取 .env）
python face_dedup_pipeline.py videos/video.mp4 --cuda
```

**优点**：持久，一次修改，多次使用；**缺点**：需要手动编辑文件

#### 方式 3️⃣：修改 config.py（推荐用于项目级配置）
```python
# 编辑 config.py
DEDUP_THRESHOLD = 0.9
YAW_THRESHOLD = 10
BATCH_SIZE = 64
```

然后运行脚本：
```bash
python face_dedup_pipeline.py videos/video.mp4 --cuda
```

**优点**：代码级配置，最灵活；**缺点**：需要改动源代码

### 配置优先级

```
命令行参数 > .env 环境变量 > config.py 默认值
     ↓           ↓              ↓
   最高         中等            最低
```

**说明**：优先级高的配置会覆盖低的。例如，如果同时指定了 `--threshold 0.9` 和 `.env` 中的 `DEDUP_THRESHOLD=0.8`，则使用命令行参数 0.9。



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
