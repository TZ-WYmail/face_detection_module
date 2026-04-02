# face_detection
基于 YOLOv11 的视频人脸检测工具，能够从视频文件中自动检测、提取并保存人脸图像，同时记录人脸出现的时间戳信息。该工具采用批处理方式高效处理视频帧，支持 GPU 加速，适用于视频内容分析、人脸识别前处理等场景。
## 功能特点

- **高效人脸检测**：使用 YOLOv11 模型进行快速准确的人脸检测
- **批量处理**：采用批处理方式高效处理视频帧
- **GPU 加速**：支持 CUDA 加速，提高处理速度
- **时间戳记录**：记录每个人脸出现的精确时间
- **进度可视化**：使用进度条显示处理进度
- **结果导出**：将检测结果保存为图像文件和 CSV 格式的记录

## 安装说明

### 环境要求

- Python 3.8 或更高版本（推荐 3.9/3.10）
- 若需 GPU 加速：安装支持的 CUDA 与对应版本的 PyTorch

### 依赖库（建议在虚拟环境/conda 环境中安装）

推荐先根据系统与 CUDA 版本通过 PyTorch 官方安装页面选择合适的安装命令，示例：

```bash
# (conda 方式示例)
conda create -n face python=3.9 -y
conda activate face
# 请按 https://pytorch.org/ 指定适合你的 CUDA 版本的安装命令，例如：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 然后安装其余依赖
pip install opencv-python ultralytics tqdm numpy
```

或者纯 pip 环境下（无 CUDA 或已按需安装 PyTorch）：

```bash
pip install opencv-python ultralytics tqdm numpy
# 如果未安装 PyTorch，请按系统与 CUDA 版本安装 torch
```

注意：
- `ultralytics` 版本与 PyTorch 兼容性可能影响模型加载；出现问题时请参考 `ultralytics` 文档。
- 若在无 GUI 的服务器上运行，考虑安装 `opencv-python-headless` 代替 `opencv-python`。

### 模型文件

项目需要 YOLOv11 人脸检测模型文件 `yolov11n-face.pt`，请确保该文件位于项目根目录或在脚本中使用绝对路径指向模型文件。

## 使用方法

1. 将待处理的视频文件放置在项目目录下，默认文件名为 `1.mp4`
2. 运行主程序：

```bash
python scripts/face_detection/face_detection.py <video_or_directory> [--output-dir OUT] [--recursive]
```

3. 程序将自动处理视频并在 `detected_faces` 目录下保存检测到的人脸图像
4. 程序将为每个处理的视频在输出目录下创建单独子目录（默认 `detected_faces/<video_stem>/`），并在该目录下保存检测到的人脸图像
5. 每个视频的检测记录会保存在输出目录下的 `{video_stem}_face_records.txt` 文件中

## 输出说明

### 人脸图像

检测到的人脸图像将保存在 `detected_faces` 目录下，文件名格式为：
```
face_{序号:04d}_time_{分钟}_{秒}.jpg
```

### 记录文件

`face_records.txt` 文件包含以下信息：
- 人脸ID：检测到的人脸序号
- 时间(MM:SS)：人脸出现的分钟和秒数
- 时间戳(秒)：从视频开始计算的秒数
- 置信度：检测的置信度分数
- 图片路径：对应人脸图像的保存路径

## 参数调整

脚本现在使用命令行参数控制行为，示例：

```bash
# 处理单个视频，保存到默认输出目录
python scripts/face_detection/face_detection.py video/video-1.mp4

# 处理目录中的所有视频（非递归），输出到 custom_out
python scripts/face_detection/face_detection.py video/video-1.mp4 --output-dir custom_out

# 递归查找目录下所有视频，并自定义参数
python scripts/face_detection/face_detection.py /path/to/videos --recursive --output-dir results --batch-size 16 --sample-interval 4 --confidence 0.6
```

可用参数：

- `input`：必填，视频文件路径或包含视频的目录
- `--output-dir, -o`：输出目录（默认: `detected_faces`）
- `--batch-size`：批处理大小（默认: 8）
- `--sample-interval`：采样间隔（帧）（默认: 8）
- `--confidence`：检测置信度阈值（默认: 0.5）
- `--recursive`：若为目录，是否递归查找子目录中的视频

## ByteTrack 去重脚本（单文件版）

项目同时提供单文件脚本 `scripts/face_detection/byte_track_dedup_preview.py`，该脚本基于 YOLO 检测结果并使用 ByteTrack 进行轨迹关联，从而实现去重与可视化预览。主要特点：

- 支持三种保存策略：`first`（首次出现保存）、`best`（全程覆盖保存最佳）和 `best_end`（轨迹结束时保存最佳）
- 支持对单个视频或目录批量处理（可递归）并为每个视频创建独立输出子目录
- 生成可视化预览视频并在输出目录内写入多种记录文件（首帧/best/best_end）

运行示例：

```bash
# 单视频 - track 模式，使用 first 保存策略
python scripts/face_detection/byte_track_dedup_preview.py video/video-1.mp4 --mode track --save-strategy best --output-dir results

# 处理目录内所有视频，递归查找，并使用 best_end 策略
python scripts/face_detection/byte_track_dedup_preview.py /path/to/videos --recursive --mode track --save-strategy best_end --output-dir results

# 仅检测模式（不使用 ByteTrack）
python scripts/face_detection/byte_track_dedup_preview.py video/video-1.mp4 --mode detect --batch-size 16 --sample-interval 4
```

额外依赖：若要使用 ByteTrack 去重功能，需要安装 YOLOX 的跟踪模块或相应的 ByteTrack 实现；在无 ByteTrack 的环境下，脚本会自动回退为纯检测模式。

安装建议：

- 方式 A（推荐，先安装 PyTorch）：

```bash
# 1) 安装/激活虚拟环境（可选）
python -m venv .venv && source .venv/bin/activate

# 2) 安装 PyTorch（请按你的 CUDA 版本从 https://pytorch.org/ 选择命令）
# 示例（CPU 或无 CUDA）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3) 安装 yolox（包含跟踪模块）
pip install yolox

# 4) 安装 ByteTrack（如果需要更完整/官方实现，可从源码安装）
pip install pytracking bytetrack
```

- 方式 B（从源码安装 YOLOX 与 ByteTrack，适用于需要最新代码或自定义构建）：

```bash
# YOLOX
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -r requirements.txt
pip install -v -e .

# ByteTrack（ifzhang/ByteTrack）
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
pip install -v -e .
```

说明：不同系统/环境下包名与安装方式可能存在差异（例如部分系统提供 `bytetrack` 的预编译包）。若脚本在运行时提示 `ModuleNotFoundError: No module named 'yolox'`，请按上述步骤安装 YOLOX 或使用源码方式确保 `yolox.tracker.byte_tracker` 可导入。

## 技术实现

- 使用 OpenCV 读取视频帧
- 采用 YOLOv11 模型进行人脸检测
- 使用 PyTorch 进行模型推理
- 通过批处理方式提高处理效率
- 使用多线程优化性能

## 注意事项

- 处理大型视频文件可能需要较长时间
- GPU 加速需要安装 CUDA 和对应版本的 PyTorch
- 检测效果受视频质量、人脸角度、光线等因素影响

## 许可证

[MIT License](https://opensource.org/licenses/MIT)
