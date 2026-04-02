# 人脸去重日志指南

## 概述

现已为去重流程添加**详细的日志输出**，记录每次人脸匹配和新增的操作。这使得用户可以实时追踪去重的决策过程。

## 日志层级

### 📍 INFO级别日志（主要信息）
这些日志记录每次成功的去重操作：

#### 🔗 去重匹配日志
当检测到**重复的人脸**（相似度足够高）时输出：

```
INFO - face_dedup_utils - 🔗 去重匹配: 检测到重复人脸 | 现有ID=00001 | 相似度=0.7324 | 阈值=0.5
```

**含义**：
- 检测到新的人脸特征与现有人脸ID=00001的相似度为0.7324
- 因为0.7324 ≥ 0.5（阈值），判定为同一个人，不创建新ID
- 这个人的出现次数+1

#### ✨ 新建人脸ID日志
当检测到**新的不同人脸**时输出：

```
INFO - face_dedup_utils - ✨ 新建人脸ID: 00002 | 总数=2 | 头部姿态(Y=-3.5°, P=2.1°, R=1.2°)
```

**含义**：
- 创建了新的人脸ID=00002
- 目前数据库中共有2个不同的人
- 该人脸的头部姿态信息（偏航角、俯仰角、翻滚角）

### 🐛 DEBUG级别日志（诊断信息）

这些日志帮助诊断为什么某些人脸未被匹配：

```
DEBUG - face_dedup_utils - ❌ 去重未匹配: 相似度不足 | 最高相似度=0.4823 | 阈值=0.5
```

**含义**：
- 新的人脸特征与最相似的现有人脸相似度为0.4823
- 因为0.4823 < 0.5（阈值），判定为不同的人
- 会创建新的ID

#### 📹 处理上下文日志
处理每个人脸/轨迹前的上下文信息：

```
DEBUG - face_dedup_pipeline - 📹 处理轨迹ID=5 | 共12帧 | 质量=0.8956
```

**含义**：
- 正在处理的轨迹（连续检测）的ID=5
- 该轨迹中捕获了12帧人脸图像
- 最佳人脸的质量分数=0.8956

## 使用方法

### 1. 查看完整日志信息
默认输出为 INFO 级别，只显示匹配和新增的操作。

运行命令：
```bash
python face_dedup_pipeline.py videos/your_video.mp4 --detector insightface
```

输出示例：
```
2024-01-15 10:30:45 - face_dedup_pipeline - INFO - ✅ 检测到 GPU: NVIDIA GeForce RTX 3090
2024-01-15 10:30:45 - face_dedup_pipeline - INFO - 处理视频: videos/video-2.mp4
2024-01-15 10:30:46 - face_dedup_utils - INFO - ✨ 新建人脸ID: 00001 | 总数=1 | 头部姿态(Y=2.3°, P=-1.2°, R=0.5°)
2024-01-15 10:30:47 - face_dedup_utils - INFO - 🔗 去重匹配: 检测到重复人脸 | 现有ID=00001 | 相似度=0.7856 | 阈值=0.5
2024-01-15 10:30:48 - face_dedup_utils - INFO - ✨ 新建人脸ID: 00002 | 总数=2 | 头部姿态(Y=-5.1°, P=0.0°, R=-1.5°)
2024-01-15 10:30:49 - face_dedup_utils - INFO - 🔗 去重匹配: 检测到重复人脸 | 现有ID=00002 | 相似度=0.8234 | 阈值=0.5
```

### 2. 查看调试信息（包括未匹配的原因）
设置日志级别为 DEBUG：

```bash
python face_dedup_pipeline.py videos/your_video.mp4 --detector insightface --log-level DEBUG
```

或在代码中添加：
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

输出示例：
```
2024-01-15 10:30:46 - face_dedup_pipeline - DEBUG - 📹 处理轨迹ID=1 | 共5帧 | 质量=0.9123
2024-01-15 10:30:46 - face_dedup_utils - DEBUG - ❌ 去重未匹配: 相似度不足 | 最高相似度=0.4532 | 阈值=0.5
2024-01-15 10:30:46 - face_dedup_utils - INFO - ✨ 新建人脸ID: 00001 | 总数=1 | 头部姿态(Y=1.2°, P=-2.3°, R=0.1°)
2024-01-15 10:30:48 - face_dedup_pipeline - DEBUG - 📹 处理轨迹ID=2 | 共8帧 | 质量=0.8765
2024-01-15 10:30:48 - face_dedup_utils - DEBUG - ❌ 去重未匹配: 距离过大 | 最小距离=0.5432 | 阈值=0.5
```

### 3. 保存日志到文件

创建一个脚本 `run_with_logging.py`：

```python
import logging
from face_dedup_pipeline import main

# 配置文件日志
fh = logging.FileHandler('dedup_log.txt', encoding='utf-8')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# 获取root logger并添加文件处理器
root_logger = logging.getLogger()
root_logger.addHandler(fh)
root_logger.setLevel(logging.DEBUG)

# 运行去重流程
main()
```

运行：
```bash
python run_with_logging.py videos/your_video.mp4 --detector insightface
```

日志将同时输出到控制台和 `dedup_log.txt` 文件。

## 日志格式说明

标准日志格式：
```
时间戳 - 模块名 - 日志级别 - 日志内容
2024-01-15 10:30:45 - face_dedup_utils - INFO - ✨ 新建人脸ID: 00001 | 总数=1 | 头部姿态(Y=2.3°, P=-1.2°, R=0.5°)
```

## 参数说明

### 去重阈值 (`--threshold`)
- **默认值**: 0.6
- **范围**: 0.0 - 1.0
- 更低的阈值 → 更容易判定为同一个人（可能误认)
- 更高的阈值 → 更严格，可能将同一个人判定为不同人

调整示例：
```bash
# 严格去重（更不容易合并）
python face_dedup_pipeline.py videos/video.mp4 --threshold 0.7

# 宽松去重（更容易合并）
python face_dedup_pipeline.py videos/video.mp4 --threshold 0.5
```

对应的日志会显示实际使用的阈值：
```
INFO - 🔗 去重匹配: 检测到重复人脸 | 现有ID=00001 | 相似度=0.7324 | 阈值=0.7
```

## 常见问题分析

### Q: 为什么很多人脸都被判定为重复？
检查日志中的相似度分数和阈值。如果相似度都很高（> 0.8），可能：
- 阈值设置过低
- 所有视频片段都是同一个人

**解决**：提高 `--threshold` 参数

### Q: 为什么会创建过多的人脸ID？
检查日志中的相似度分数。如果都显示"相似度不足"，可能：
- 阈值设置过高
- 真的是多个不同的人

**解决**：降低 `--threshold` 参数或检查输入视频

### Q: 相似度的计算方法是什么？
- **默认**: 余弦相似度 (cosine similarity)
  - 范围: -1.0 到 1.0，通常看到 0.4 到 1.0
  - 高于0.5表示相似，高于0.7表示非常相似
  
- **可选**: 欧几里得距离 (euclidean distance)
  - 范围: 0.0 到较大值
  - 低于阈值表示相似

## 日志示例分析

**完整的处理流程日志**：

```
INFO - 处理视频: videos/video-2.mp4
DEBUG - 📹 处理轨迹ID=1 | 共5帧 | 质量=0.8934
DEBUG - ❌ 去重未匹配: 相似度不足 | 最高相似度=0.3421 | 阈值=0.6
INFO - ✨ 新建人脸ID: 00001 | 总数=1 | 头部姿态(Y=-2.5°, P=0.0°, R=1.2°)

DEBUG - 📹 处理轨迹ID=2 | 共3帧 | 质量=0.7645
INFO - 🔗 去重匹配: 检测到重复人脸 | 现有ID=00001 | 相似度=0.7234 | 阈值=0.6

DEBUG - 📹 处理轨迹ID=3 | 共8帧 | 质量=0.9123
DEBUG - ❌ 去重未匹配: 相似度不足 | 最高相似度=0.5123 | 阈值=0.6
INFO - ✨ 新建人脸ID: 00002 | 总数=2 | 头部姿态(Y=3.1°, P=-1.5°, R=-0.3°)

完成: 2 张正脸已保存
```

**分析**：
1. 轨迹1的第一个人脸被识别为新人（ID: 00001）
2. 轨迹2的人脸与ID 00001相似度0.7234 > 0.6，判定为同一个人
3. 轨迹3的人脸相似度0.5123 < 0.6，判定为不同的人（ID: 00002）
4. 最终识别出2个不同的人

## 注意事项

1. **日志性能**: 输出日志可能会略微降低处理速度，但通常不明显
2. **日志量大**: 视频长度大，日志文件可能很大，建议保存到文件而不是只在控制台输出
3. **隐私**: 日志中包含人脸特征向量和ID映射, 请妥善处理
4. **调试**: 在面临去重问题时，启用DEBUG级别日志查看详细的决策过程

