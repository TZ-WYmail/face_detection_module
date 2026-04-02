# 使用 .pt 模型避免误检 - 快速开始

## 问题

当前系统使用 **ONNX** 模型，导致识别出：
- 🖱️ 鼠标
- 📱 手机/平板
- 🎮 遥控器
- 💻 其他物体

**原因**：ONNX 模型无法提取关键点，无法进行有效验证。

## 解决方案（推荐）

### 步骤1：下载 .pt 模型

选择以下任一方式：

**选项 A：自动下载（最简单）**
```bash
cd /home/tanzheng/Desktop/face_detection_module

# YOLO会自动下载到缓存
python -c "from ultralytics import YOLO; YOLO('yolov11n-face.pt')"
```

**选项 B：手动下载**
```bash
# 创建目录
mkdir -p models/yolo

# 下载 yolo26n-face 模型
cd models/yolo
wget https://github.com/akanametov/yolo-face/releases/download/v1.0.0/yolo26n-face.pt

# 或使用 curl
curl -L -o yolo26n-face.pt \
  https://github.com/akanametov/yolo-face/releases/download/v1.0.0/yolo26n-face.pt
```

### 步骤2：验证模型已加载

运行管道时检查日志：

```bash
python face_dedup_pipeline.py videos/video-2.mp4 \
  --detector yolo \
  --cuda \
  -o output/
```

**正确的日志**（使用 .pt）：
```
✅ 找到 .pt 人脸模型: models/yolo/yolo26n-face.pt
✅ 使用 YOLO .pt 检测器 (优：可提取关键点)
   模型: models/yolo/yolo26n-face.pt
```

**错误的日志**（仍使用 ONNX）：
```
⚠️  未找到 .pt 模型，使用 ONNX: models/yolo/yolov11n-face.onnx
⚠️  使用 YOLO .onnx 检测器 (备选：无关键点)
```

## 模型对比

| 特性 | InsightFace | YOLO .pt | YOLO ONNX |
|------|-------------|----------|-----------|
| **关键点** | ✅ 5个 | ✅ 5个 | ❌ 无 |
| **Embedding** | ✅ 有 | ❌ 无 | ❌ 无 |
| **GPU加速** | ✅ 是 | ✅ 是 | 部分 |
| **误检率** | 最低 | 很低 | 高 |
| **速度** | 快 | 快 | 中等 |
| **推荐度** | 🌟🌟🌟 | 🌟🌟 | ⚠️ |

## 现在怎么做

### 立即测试
```bash
# 使用 InsightFace（最好的选择）
python face_dedup_pipeline.py videos/video-2.mp4 \
  --detector insightface \
  --cuda \
  --threshold 0.6 \
  --yaw-threshold 15 \
  --pitch-threshold 15 \
  --roll-threshold 10 \
  -o output_insightface/
```

输出中应该**看不到**鼠标、手机等非人脸物体。

### 如果坚持用YOLO

使用 .pt 模型：
```bash
python face_dedup_pipeline.py videos/video-2.mp4 \
  --detector yolo \
  --cuda \
  -o output_yolo_pt/
```

如果没有 .pt 模型，系统会：
1. 尝试 ONNX 模型（有误检）
2. **自动启用严格分类** ⭐
   - 检查皮肤色调
   - 检查纹理复杂度
   - 检查边缘密度
   - 检查宽高比

## 验证严格分类是否生效

看日志中是否有：
```
⚠️  检测到非人脸对象 (轨迹X): 皮肤色比例过低 + 纹理异常 + ...
```

这表示系统成功过滤了误检。

## 文件位置

下载后的 .pt 模型应该放在：
```
face_detection_module/
├── models/
│   └── yolo/
│       ├── yolo26n-face.pt      ← 你下载的 .pt 模型放这里
│       └── yolov11n-face.onnx   ← 原有的 ONNX 模型
├── videos/
├── output/
└── ...
```

## 常见问题

**Q: 下载 .pt 会很慢吗？**
A: 取决于网络速度。yolo26n-face.pt 约 6-8 MB，normally 很快。

**Q: 可以同时有多个 .pt 模型吗？**
A: 可以。系统会按优先级选择第一个找到的：
   1. yolov11n-face.pt
   2. yolov11s-face.pt  
   3. yolo26n-face.pt
   4. yolov10n-face.pt

**Q: 下载后还要配置什么吗？**
A: 不需要。系统会自动检测并使用。

**Q: InsightFace 和 YOLO 哪个更好？**
A: InsightFace 更好（有 embedding），但 YOLO .pt 也很不错。

## 后续改进监控

跑完视频后，检查输出文件：

```bash
# 查看保存了多少人脸
ls -la output/embeddings/ | wc -l

# 查看有无误检的人脸
# （如果内容都是正确的人脸，说明分类成功）
```

## 总结

| 当前使用 | 推荐切换到 | 命令 |
|---------|----------|------|
| ONNX无验证 | InsightFace | `--detector insightface` |
| ONNX无验证 | YOLO .pt | 下载 .pt，然后 `--detector yolo` |
| 继续用ONNX | - | 自动启用严格分类✅ |

---

**建议**：立即下载 .pt 模型或使用 InsightFace，避免依赖严格分类这个"补救措施"。

