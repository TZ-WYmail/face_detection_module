# 人脸保存质量改进指南

## 概述

本文档描述了如何以美观的方式保存人脸，避免变形，并在保存后进行质量验证。

## 新增功能

### 1. `save_face_pretty()` - 美观人脸保存

**位置**: `face_dedup_utils.py::590-656`

#### 功能
- 自动计算人脸周围的padding，保留面部环境
- 如果提供了关键点，自动旋转人脸使其水平
- 保持原始宽高比，避免压缩变形
- 使用立方插值保证图像质量
- 白色背景铺底，使人脸更加突出

#### 参数
```python
save_face_pretty(
    img: np.ndarray,           # 输入图像
    bbox: Tuple[int, int, int, int],  # 人脸bbox (x1, y1, x2, y2)
    kps: Optional[np.ndarray] = None, # 人脸关键点（可选）
    output_size: int = 224     # 输出图像大小（保持宽高比）
)
```

#### 使用示例
```python
# 基本使用
pretty_face = save_face_pretty(frame, bbox=(100, 50, 200, 250))

# 带关键点的自动旋转校正
pretty_face = save_face_pretty(frame, bbox, kps, output_size=256)
```

#### 输出
- 结果是边长为 `output_size` 的正方形图像
- 人脸居中放置，周围为浅灰色背景
- 人脸自动旋转使眼睛水平

### 2. `FaceSaveQualityValidator` - 质量检验器

**位置**: `face_dedup_utils.py::659-764`

#### 功能
检验保存的人脸图像是否符合质量标准：

| 检验项 | 默认值 | 目的 |
|-------|--------|------|
| 最小宽度 | 50px | 避免过小的人脸 |
| 最小高度 | 50px | 避免过小的人脸 |
| 最大宽高比 | 1.5 | 避免过度变形 |
| 最小亮度 | 50 | 避免图像过暗 |
| 最大亮度 | 200 | 避免图像过曝 |
| 最小对比度 | 0.15 | 避免严重模糊 |

#### 使用示例
```python
# 创建验证器
validator = FaceSaveQualityValidator(
    min_face_width=40,
    min_face_height=40,
    max_aspect_ratio=1.5,
    min_brightness=50,
    max_brightness=200,
    min_contrast=0.15
)

# 验证图像
is_valid, reason = validator.validate(face_img, "path/to/face.jpg")
if not is_valid:
    print(f"质量检验失败: {reason}")
else:
    print("质量检验通过")
```

#### 验证结果
- 输出: `(is_valid: bool, reason: str)`
- 所有检验项都通过时返回 `(True, "通过")`
- 任何一项失败时返回 `(False, failure_reason)`

### 3. `save_face_with_validation()` - 综合保存函数

**位置**: `face_dedup_pipeline.py::114-168`

#### 功能
一站式解决方案，集成了美观保存和质量检验：

1. 使用 `save_face_pretty()` 美观保存人脸
2. 自动进行质量检验
3. 质量检验失败时自动删除不符合标准的图像
4. 记录详细的日志信息

#### 参数
```python
save_face_with_validation(
    face_img: np.ndarray,      # 人脸区域图像
    bbox: Tuple[int, int, int, int],  # 边界框
    kps: Optional[np.ndarray], # 关键点
    img_path: str,             # 保存路径
    validator: FaceSaveQualityValidator,  # 检验器
    adjust_for_quality: bool = True  # 使用美观保存
)
```

#### 返回值
- `True`: 保存成功且通过质量检验
- `False`: 保存失败或质量检验未通过

#### 工作流程
```
输入人脸图像
  ↓
美观保存→ 白色背景上的正方形图像
  ↓
质量检验
  ↓
检验通过？
  ├─ 是 → 保存成功 ✅
  └─ 否 → 删除文件 ❌
```

#### 使用示例
```python
# 在process_frame_with_tracking中
if save_face_with_validation(
    face_crop,
    (x1, y1, x2, y2),  # bbox
    kps,               # 关键点
    img_path,
    self.quality_validator,
    adjust_for_quality=True  # 启用美观保存
) and save_embedding(emb, emb_path):
    # 保存成功
    saved_count += 1
```

## 集成情况

### FrontalFaceExtractor 类的改进

在 `face_dedup_pipeline.py` 中，`FrontalFaceExtractor` 的 `__init__` 方法现已包含：

```python
# 质量检验器（用于保存后的人脸图像质量检查）
self.quality_validator = FaceSaveQualityValidator(
    min_face_width=40,
    min_face_height=40,
    max_aspect_ratio=1.5,
    min_brightness=50,
    max_brightness=200,
    min_contrast=0.15
)
```

### 保存位置的更新

| 位置 | 方法 | 集成 |
|------|------|------|
| process_frame_with_tracking (first策略) | `save_face_with_validation()` | ✅ 完全集成 |
| process_ended_tracks (best策略) | `quality_validator.validate()` | ✅ 完全集成 |
| process_video (最终track处理) | `quality_validator.validate()` | ✅ 完全集成 |

## 质量检验日志

### 检验通过的日志
```
✅ 质量检验通过 (face_00001.jpg): 尺寸=224x224, 亮度=125.3, 对比度=0.421
```

### 检验失败的日志
```
⚠️  质量检验失败 (face_00001.jpg): 人脸尺寸过小: 35x40 (最小: 50x50)
⚠️  质量检验失败 (face_00002.jpg): 人脸亮度异常: 35.2 (应在 50-200)
⚠️  质量检验失败 (face_00003.jpg): 人脸对比度过低: 0.085 (最小: 0.15)
```

## 美观保存的优势

### 避免变形
- **传统方法**: 直接缩放bbox区域，可能导致宽高比失真
- **新方法**: 保持原始宽高比，使用白色背景铺底

### 自动姿态校正
- 如果提供关键点，自动旋转使眼睛水平
- 消除轻微的头部倾斜

### 背景保留
- 在原bbox基础上扩展15%的padding
- 保留面部周围环境，更加自然美观
- 有助于后续的face recognition任务

### 质量保证
- 立方插值保证缩放质量
- 保存后立即验证是否符合标准
- 不符合标准的图像自动删除

## 配置示例

### 严格模式（用于高精度应用）
```python
validator = FaceSaveQualityValidator(
    min_face_width=100,
    min_face_height=100,
    max_aspect_ratio=1.2,    # 更严格
    min_brightness=80,
    max_brightness=180,
    min_contrast=0.25        # 更严格
)
```

### 宽松模式（用于快速处理）
```python
validator = FaceSaveQualityValidator(
    min_face_width=30,
    min_face_height=30,
    max_aspect_ratio=2.0,    # 允许更大变形
    min_brightness=40,
    max_brightness=220,
    min_contrast=0.10        # 允许更低对比度
)
```

## 性能考虑

### 计算成本
- `save_face_pretty()`: 主要成本在旋转变换，约 1-2ms 
- `FaceSaveQualityValidator.validate()`: 约 0.5-1ms（灰度转换 + 统计计算）
- 总体影响: 每张人脸增加 1.5-3ms

### 磁盘 I/O 影响
- 不符合标准的人脸会被删除，减少磁盘占用
- 整体不符合标准的人脸通常占 5-15%

### 内存占用
- `FaceSaveQualityValidator` 对象化简，内存占用忽略不计
- 单张人脸处理过程中内存占用不增加

## 故障排除

### 问题: 大多数人脸都被标记为质量检验失败

**解决方案**:
1. 检查输入视频质量
2. 调整 `min_contrast` 参数（如果图像模糊）
3. 调整 `min_brightness` 参数（如果图像过暗）
4. 使用宽松模式配置

### 问题: 人脸被过度变形

**解决方案**:
1. 检查 `save_face_pretty()` 的参数
2. 如果不需要自动旋转，传 `kps=None`
3. 增加 `output_size` 参数值

### 问题: 某些正确的人脸被删除

**解决方案**:
1. 调整质量检验参数
2. 使用 `validator.validate(img, "debug")` 来查看详细原因
3. 根据具体应用调整阈值

## 最佳实践

1. **定期审查保存的人脸**: 检查是否有遗漏或错误删除
2. **调整参数适应数据集**: 不同的视频可能需要不同的参数
3. **监控日志**: 定期检查质量检验日志找出问题
4. **保存原始数据**: 可考虑保留备份，便于重新处理

## 后续改进方向

1. **基于BRISQUE的质量评分**: 使用图像质量指标代替简单参数
2. **自适应参数调整**: 根据检测到的人脸特性动态调整参数
3. **人脸识别度评估**: 使用人脸识别模型评估保存后的人脸可识别性
4. **多尺度保存**: 同时保存多种大小的人脸供不同用途使用

