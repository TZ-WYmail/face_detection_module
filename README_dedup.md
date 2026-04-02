高质量人脸去重流水线说明

简介
--
本目录新增两个文件：
- `face_dedup_utils.py`：检测/对齐/轻量跟踪/embedding 回退与去重工具。
- `face_dedup_pipeline.py`：命令行入口，用于从单个视频或目录批量提取高质量人脸并去重（建议先在小样本上试验参数）。

快速运行
--
示例：
```bash
python scripts/face_detection/face_dedup_pipeline.py video/video-1.mp4 --output-dir results/dedup --use-tracks --save-strategy best_end --require-quality --threshold 0.6
```

扩展说明
--
- 如果环境安装了 `insightface`，流水线会优先使用它（带关键点与 embedding）；否则回退到 YOLO 或 OpenCV Haar。
- 推荐先使用 `--require-quality` 打开五官质量过滤，再根据结果放宽。
- `--threshold` 对应 dedup 的相似度阈值（cosine），可调节以获得更高召回或更低误合并。

后续合并
--
此实现独立于主流程，待你确认去重质量后我可以把逻辑合并入 `memory_system` 的人脸处理链路中（建议把 `Detector` 的初始化托管到 `FaceFeatureExtractor`，并把 dedup 结果映射到 `PersonaMemory`）。

处理流程详解
--
下面为 `face_dedup_pipeline.py` + `face_dedup_utils.py` 当前实现的逐步处理逻辑（按代码实际行为精确描述）。

1. 初始化
	- 解析命令行参数并创建每个视频的输出子目录（默认根目录由 `--output-dir` 指定）。
	- 根据 `--detector`（`auto`/`insightface`/`yolo`/`haar`）初始化 `Detector`：优先使用 `insightface`（若安装且可用，会同时提供关键点与 embedding）；否则依次回退到 `ultralytics.YOLO` 或 OpenCV Haar。
	- 如果启用跟踪（`--use-tracks`），创建轻量 `IoUTracker`（参数 `--iou`、`--max-age` 控制匹配阈值和轨迹允许丢失帧数）。
	- 初始化 `Deduper`（基于余弦或欧氏度量，阈值由 `--threshold` 控制）用于库内最近邻匹配。

2. 视频读取与抽帧
	- 使用 OpenCV 打开视频并读取 FPS 与总帧数。
	- 以 `--sample-interval` 决定抽样间隔（默认每帧或按采样率fallback），仅对抽样帧进行检测与去重处理；同时写 `preview.mp4`（用于可视化）。

3. 单帧处理（每个采样帧）
	- 检测：调用 `Detector.detect(frame)`，返回若干检测项，每项包含 `bbox`、可选 `kps`（关键点）、`confidence` 与可选 `embedding`（若使用 InsightFace 则可直接获取）。
	- 质量过滤（可选）：若启用了 `--require-quality`，对每个检测项调用 `is_high_quality_face()`，该函数基于左右眼与鼻尖关键点进行判断（两眼间距、鼻尖相对偏移等），严格过滤极端侧脸、远景或遮挡较严重的人脸。若关键点缺失则按严格默认丢弃。
	- 对齐：对通过质量过滤的检测用 `align_face()` 基于左右眼做仿射对齐（输出固定尺寸，如 112x112），保证 embedding 提取的角度一致性。
	- Embedding 提取：优先使用检测器自带的 embedding（InsightFace）；若无，则使用 `SimpleEmbedder` 的轻量回退（HSV 三通道直方图归一化），供去重使用（回退 embedding 精度较低，仅作弱匹配）。

4. 跟踪与聚合（启用 `--use-tracks` 时）
	- 把本帧检测的 bbox 列表传入 `IoUTracker.update()`，该实现采用贪婪 IoU 矩阵匹配完成与已有轨迹的关联，并在必要时创建新轨迹。
	- 对每个活跃轨迹（track）：找到与之 IoU 最大的检测，更新轨迹的 `embs`（追加当前帧 embedding）、`best_conf`（保存最高置信度帧用于 `best_end`）等聚合信息。
	- 保存策略：
	  - `first`：在轨迹第一次出现并且 embedding 在去重库中未命中时，立即把当前帧保存为该 persona 的样本并把 embedding 入库（避免后续重复保存）。
	  - `best_end`：在轨迹结束时（轨迹从上一帧活跃到当前帧消失）计算轨迹内 embedding 的均值作为聚合 embedding，先在去重库查找匹配；若无匹配则入库并把在轨迹期间记录到的 `best_img`（最高置信度帧）保存为代表样本。
	  - `best`：当前实现对 `best` 的支持较弱（主要以 `best_end` 为主）；如需实时覆盖保存最佳帧，可在后续迭代中把轨迹内 `best_img` 在每帧检查并替换/写盘。

5. 无跟踪模式（`--no-tracks`）
	- 对每个检测项独立做质量过滤、对齐与 embedding 提取。
	- 使用 `Deduper.find_match()` 在已保存 embedding 库内查找最相似者：
	  - 若命中（相似度 >= `--threshold`），视为同一 persona，不另存图片；
	  - 若未命中且 embedding 可用，则 `Deduper.add()` 将 embedding 加入库并把当前裁剪图像保存为新 persona 的代表样本（`first` 行为）。

6. 去重（Deduper）实现要点
	- 默认使用余弦相似度（`--metric cosine`）；`Deduper` 会把 embedding 归一化并维护内存中的 embedding 列表与对应 persona_id。
	- 在 track 结束时（`best_end`），对轨迹内所有 embedding 做平均（作为轨迹聚合 embedding），再与去重库比对并决定保存/合并。

7. 输出与记录
	- 保存图片：按 persona id 命名（例如 `face_{pid:05d}_track{tid}_BEST_END.jpg` 或 `face_{pid:05d}_f{frame}.jpg`），保存在每个视频的输出子目录下。
	- 记录文件：`face_records_dedup.txt`（CSV），每行包含 `persona_id,track_id,time,timestamp,confidence,image_path,frame_count`。
	- 预览视频：`preview.mp4`，用于可视化处理结果（原始帧或可扩展为带框预览）。

8. 关键参数与调优建议
	- `--threshold`（默认 0.6）：Deduper 的相似度阈值，越高合并越谨慎（误合并下降，漏判上升）。
	- `--require-quality`：强推荐初次打开，能直接过滤大量侧脸/遮挡误检，显著提高入库质量。
	- `--iou` / `--max-age`：影响 IoUTracker 的跟踪稳定性；在低帧率或快速运动场景下适当减小 `max_age` 或提高 `iou`。
	- 若可用，优先安装 `insightface + onnxruntime(-gpu)`：它提供高质量的 5 点关键点对齐与 ArcFace embedding，可把去重精度提升一个档次。

9. 局限与未来改进
	- 回退 embedding（HSV 直方图）鲁棒性有限，建议在目标平台安装 `insightface` 或其他 ArcFace 实现以获得稳定的人脸特征。
	- 当前 `best` 策略为后续扩展项；若希望在轨迹运行时持续维护并覆盖最佳帧，可把轨迹的 `best_img` 在每帧写盘或按条件替换已有样本。
	- 可选增强：把外观 ReID（embedding 距离）与运动预测（Kalman）融合到关联步骤，或在轨迹层加入短时 ID 合并策略以降低 ID switch。

10. 与主流程合并建议
	 - 将 `Detector` 的初始化和 embedding 提取逻辑移入 `memory_system/perception/face_feature_extractor.py`，把 `Deduper` 的匹配结果映射为 `PersonaMemory` 的 persona_id（而非单独文件名命名空间）。
	 - 在 `MemoryBuilder` 中调用去重模块时，优先传入 `require_quality` 筛选后的结果，确保写入的 persona 特征满足质量标准。

如果你同意这份描述，我会把它追加到项目总 README 或把去重模块按需合并进 `memory_system`。如需我现在把 `best` 策略补全为“实时覆盖最佳帧”，我也可以继续实现并在小视频上跑测。
