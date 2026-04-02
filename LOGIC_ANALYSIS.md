# 处理流程逻辑分析报告

## 1. 用户期望的理想处理流程

```
【每帧处理】
  ↓
[第一步] InsightFace 检测 + 姿态判定
  ├─ 检测人脸 → 获取 bbox + kps + embedding
  ├─ 姿态估计 → 计算 yaw/pitch/roll
  ├─ 质量评估 → 初步过滤不合格人脸
  └─ 输出：质量合格的人脸列表
  ↓
[第二步] Embedding 相似度判断（去重）
  ├─ 对本帧每个人脸计算 embedding
  ├─ 与已有人脸库对比
  ├─ 相似度高于阈值 → 跳过（重复）
  ├─ 相似度低于阈值 → 新人脸（保留）
  └─ 输出：去重后的人脸列表
  ↓
[第三步] YOLO 跟踪（Track）
  ├─ 对去重后的人脸做跟踪
  ├─ 维护跟踪ID，关联同一人的多次出现
  └─ 输出：带跟踪ID的人脸
  ↓
[第四步] 连续重复过滤
  ├─ 检测同一 track id 在连续帧内重复
  ├─ 保留质量最好的一帧
  ├─ 跳过其他重复帧
  └─ 输出：已过滤的人脸记录

最终保存：persona_id + track_id + 图像 + embedding
```

---

## 2. 当前实现的处理流程

### 分支 A：使用 DetectionTracker（默认，`--use-detection-tracker=True`）

```
【每帧处理】
  ↓
[顺序 1] DetectionTracker 跟踪（轻量级）
  ├─ self.det.detect(frame) → 检测人脸（InsightFace 或 YOLO ONNX）
  ├─ 获得 bbox + kps + confidence
  ├─ detection_tracker.update() → 中心距离+尺寸匹配
  ├─ 分配 track id（本地状态机）
  └─ 输出：(xyxy, confs, ids)
  ↓
[顺序 2] 逐个处理 tracked box
  ├─ 对每个 box 评估质量（姿态判定）
  ├─ 如果是正脸 → 对齐、提取 embedding
  ├─ 去重匹配：self.deduper.find_match(embedding)
  ├─ 如果去重失败 → 新 persona → 保存
  └─ 输出：保存的人脸文件
  ↓
[顺序 3] 帧级去重（可选）
  └─ 如果本帧多个正脸相似度高 → 保留质量最好的
```

**问题：质量判定发生在跟踪之后**

### 分支 B：使用 ByteTrack（若 `--no-detection-tracker`）

```
【每帧处理】
  ↓
[顺序 1] ByteTrack 跟踪（Ultralytics）
  ├─ model_track.track(frame) → YOLO 内部检测+跟踪
  ├─ 获得 bbox + confidence + track_id
  └─ 输出：(xyxy, confs, ids)
  ↓
[顺序 2] 再次检测以获取 kps
  ├─ self.det.detect(frame) → 再做一次检测
  ├─ 构建 bbox → kps 查找表
  └─ 输出：bbox_to_kps 映射
  ↓
[顺序 3] 逐个处理 tracked box
  ├─ 查表获得 kps
  ├─ 评估质量
  ├─ 去重匹配
  └─ 保存
```

**问题：做了两次检测（浪费），质量判定仍在跟踪之后**

---

## 3. 关键差异与问题

| 方面 | 用户期望 | 当前实现 | 问题 |
|------|---------|---------|------|
| **质量判定时机** | 在跟踪之前 | 在跟踪之后 | 会跟踪低质人脸，浪费计算 |
| **去重判定时机** | 第二步独立执行 | 在保存时才做 | 没有"过滤已重复人脸"的机制 |
| **YOLO 角色** | 仅负责跟踪 | 可能用于检测或跟踪 | 不清晰，可能混淆 |
| **连续重复过滤** | 明确需要 | 无此机制 | 同一 track 多帧都会保存 |
| **embeddin 用途** | 先做去重判断 | 后用于匹配 persona | 时间顺序不对 |

---

## 4. 当前代码中的具体位置与表现

### 问题 1：质量判定在跟踪之后

**位置：** `process_frame_with_tracking()` 第 434-516 行
```python
# 当前顺序：
1. if getattr(self, 'detection_tracker', None) is not None:
2.     frame_detections = self.det.detect(frame)  ← 检测
3.     detected_tracker.update(...)                ← 跟踪
4. # 然后才进行质量评估（后续代码）
```

**表现：** 所有检测到的人脸都会被分配 track id，即使质量很差。

### 问题 2：缺乏"质量过滤后再跟踪"的流程

**位置：** 没有一个地方说"先过滤不合格人脸，再对合格的做跟踪"

**当前做法：** 跟踪所有人脸，然后在 `if quality_result.is_frontal` 时才选择性处理。

### 问题 3：Embedding 去重与跟踪混淆

**位置：** `process_frame_with_tracking()` 第 658-690 行
```python
# 当前做法：
pid = self.deduper.find_match(emb)  # 去重（基于 embedding）
if pid is None:
    pid = self.deduper.add(emb)      # 新人脸
    # 保存...
```

**问题：** 去重和保存耦合在一起。用户期望先去重判定，再决定是否需要跟踪。

### 问题 4：无连续重复过滤

**位置：** 整个代码没有检查"同一 track id 在连续 N 帧内是否重复"

**表现示例：**
```
Frame 0: track_123 出现 → 保存
Frame 1: track_123 出现 → 保存（重复）
Frame 2: track_123 出现 → 保存（重复）
```

应该只保存一个，或质量最好的那个。

---

## 5. 理想改进方案

### 架构建议：三层管道 (Pipeline)

```
Layer 1: 质量层 (Quality Layer)
  └─ InsightFace detect → 姿态+质量评估 → 过滤不合格 → Quality-aware 人脸列表

Layer 2: 去重层 (Dedup Layer)
  └─ Embedding 相似度判断 → 跳过重复 → Unique-person 人脸列表

Layer 3: 跟踪层 (Tracking Layer)
  └─ YOLO ByteTrack → 分配 track id → 连续重复过滤 → Final 输出
```

### 伪代码示例

```python
def process_frame_improved(frame, frame_count):
    # ===== Layer 1: 质量 =====
    all_dets = insightface_detect(frame)
    
    # 质量+姿态过滤
    quality_faces = []
    for det in all_dets:
        yaw, pitch, roll = estimate_pose(det['kps'])
        is_frontal, quality_score = self.pose_estimator.is_frontal_face(...)
        if is_frontal and quality_score > min_threshold:
            quality_faces.append({
                'bbox': det['bbox'],
                'kps': det['kps'],
                'embedding': det['embedding'],
                'quality_score': quality_score
            })
    
    if not quality_faces:
        return  # 无合格人脸
    
    # ===== Layer 2: 去重 =====
    unique_faces = []
    for face in quality_faces:
        person_id = self.deduper.find_match(face['embedding'])
        if person_id is None:
            # 新人脸
            person_id = self.deduper.add(face['embedding'])
            unique_faces.append({
                'bbox': face['bbox'],
                'kps': face['kps'],
                'embedding': face['embedding'],
                'person_id': person_id,
                'quality_score': face['quality_score']
            })
        # else: 如果去重失败，说明是重复，跳过
    
    if not unique_faces:
        return  # 全是重复人脸
    
    # ===== Layer 3: 跟踪 =====
    # 注意：YOLO track 此时只跟踪经过 Layer1 + Layer2 的人脸
    track_ids = yolo_bytetrack.track(unique_faces, frame_count)
    
    # ===== Layer 3.5: 连续重复过滤 =====
    for track_id, face in zip(track_ids, unique_faces):
        # 检查同一 track_id 在过去 N 帧是否出现
        if self.track_history.is_continuous_duplicate(track_id):
            continue  # 跳过连续重复
        
        # 保存
        self.track_history.add(track_id, face, frame_count)
        save_face(face, person_id, track_id)
```

---

## 6. 改进优先级

1. **高优先级**（直接影响保存）：
   - [ ] 实现"质量过滤后再跟踪"的管道
   - [ ] 添加连续重复过滤机制

2. **中优先级**（提升清晰度）：
   - [ ] 分离 InsightFace 检测职责（质量+去重）与 YOLO 跟踪职责
   - [ ] 明确 embedding 的用途顺序（先去重、后跟踪）

3. **低优先级**（优化性能）：
   - [ ] 避免重复检测（当前 ByteTrack 分支做两次检测）
   - [ ] 缓存 embedding 避免重复计算

---

## 7. 总结

**当前实现** 采用了"先跟踪、后质量判定"的方式，导致：
- 低质人脸也被分配 track id，浪费计算
- 缺乏明确的去重决策点
- 无连续重复过滤，同一 track 多帧都保存

**用户期望** 是"质量 → 去重 → 跟踪 → 去重" 的清晰流程，其中：
- InsightFace 专门负责质量判定 + embedding 计算
- 去重判定在跟踪之前发生
- YOLO 仅用于跟踪，不参与检测
- 连续出现的同一人脸只保留最好质量的一帧

**建议** 重构流程为分层管道，提升代码清晰度与性能。
