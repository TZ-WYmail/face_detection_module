"""face_dedup_pipeline_v2.py

改进版的人脸去重流水线

核心改进：
1. 添加头部姿态估计，只保留正脸
2. 改进去重策略，避免侧脸被识别为不同人
3. 支持多种保存策略
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import csv
import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

# 在导入 YOLO 和 ONNX 之前禁用日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow
os.environ['ONNXRUNTIME_LOGS_LEVEL'] = 'FATAL'  # ONNX Runtime
os.environ['YOLO_VERBOSE'] = 'False'  # Ultralytics YOLO
import warnings
warnings.filterwarnings('ignore')

from face_dedup_utils import (
    Detector, SimpleEmbedder, Deduper, align_face,
    evaluate_face_quality, FaceQualityResult,
    HeadPoseEstimator, save_face_pretty, FaceSaveQualityValidator,
    FaceValidityChecker, DetectionTracker, TrackHistory
)

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# GPU 检测和配置
# ============================================================================

def detect_gpu_availability():
    """
    检测GPU可用性
    
    Returns:
        bool: GPU是否可用
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA 不可用，将使用 CPU 处理")
            return False
        
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        
        logger.info(f"✅ 检测到 GPU: {device_name}")
        logger.info(f"   CUDA版本: {cuda_version}")
        logger.info(f"   设备数: {device_count}")
        logger.info(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        return True
    except ImportError:
        logger.warning("⚠️  PyTorch 未安装，无法检测 GPU")
        return False
    except Exception as e:
        logger.warning(f"⚠️  GPU 检测失败: {e}")
        return False


@dataclass
class FaceRecord:
    """人脸记录"""
    track_id: int
    embeddings: List[np.ndarray] = field(default_factory=list)
    best_quality_score: float = 0.0
    best_pose_score: float = 0.0
    best_image: Optional[np.ndarray] = None
    best_yaw: float = 0.0
    best_pitch: float = 0.0
    best_roll: float = 0.0
    first_saved: bool = False
    frontal_count: int = 0  # 正脸计数


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return cv2.imwrite(path, img)


def save_embedding(emb: np.ndarray, path: str) -> bool:
    """保存embedding向量为.npy文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        np.save(path, emb.astype(np.float32))
        return True
    except Exception as e:
        logger.error(f"保存embedding失败: {e}")
        return False


def save_face_with_validation(face_img: np.ndarray, bbox: Tuple[int, int, int, int],
                             kps: Optional[np.ndarray],
                             img_path: str, validator: FaceSaveQualityValidator,
                             adjust_for_quality: bool = True) -> bool:
    """
    保存人脸图像，使用美观的保存方式并进行质量验证
    
    Args:
        face_img: 人脸区域的原始图像
        bbox: 人脸bbox (x1, y1, x2, y2)
        kps: 人脸关键点（可选）
        img_path: 保存路径
        validator: 质量检验器
        adjust_for_quality: 是否使用美观保存方式
        
    Returns:
        是否保存成功（通过质量检验）
    """
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    
    try:
        # 选择保存方式
        if adjust_for_quality and kps is not None:
            # 使用美观的保存方式
            save_img = save_face_pretty(face_img, bbox, kps, output_size=224)
        else:
            # 使用默认的缩放方式
            save_img = face_img
        
        # 保存图像
        if not cv2.imwrite(img_path, save_img):
            logger.warning(f"⚠️  保存图像失败: {img_path}")
            return False
        
        # 保存后验证质量
        is_valid, reason = validator.validate(save_img, img_path)
        
        if not is_valid:
            logger.warning(f"⚠️  人脸质量检验失败，移除保存: {reason}")
            # 删除不符合标准的图像
            try:
                os.remove(img_path)
                logger.info(f"已删除质量不符的人脸: {img_path}")
            except Exception as e:
                logger.warning(f"无法删除文件: {e}")
            return False
        
        # 质量检验通过
        logger.info(f"✅ 人脸保存成功（已验证）: {img_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 保存人脸时出错: {e}")
        return False


def _resolve_embedding_path(embedding_path: str, rec_file_path: str, db_root: str) -> str:
    """将记录中的 embedding 路径解析为本地绝对路径。"""
    if not embedding_path:
        return ""

    p = embedding_path.strip()
    if os.path.isabs(p):
        return p

    # 支持以 './' 开头的相对路径（以项目根目录或db_root为参考）
    p_clean = p[2:] if p.startswith('./') else p
    candidate_from_cwd = os.path.abspath(p)
    if os.path.exists(candidate_from_cwd):
        return candidate_from_cwd

    candidate_from_db_root = os.path.abspath(os.path.join(db_root, p_clean))
    if os.path.exists(candidate_from_db_root):
        return candidate_from_db_root

    # 最后尝试以记录文件所在目录为基准
    return os.path.abspath(os.path.join(os.path.dirname(rec_file_path), p_clean))


def load_existing_embeddings_to_deduper(deduper: Deduper, db_root: str) -> int:
    """
    从输出目录加载历史 embedding 到 deduper，用于跨运行/跨视频匹配。

    Returns:
        加载到 deduper 的 persona 数量
    """
    if not db_root or not os.path.isdir(db_root):
        return 0

    pid_to_embeddings: Dict[int, List[np.ndarray]] = {}
    rec_files = []
    for root, _, files in os.walk(db_root):
        if 'face_records_frontal.txt' in files:
            rec_files.append(os.path.join(root, 'face_records_frontal.txt'))

    for rec_file in rec_files:
        try:
            with open(rec_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        pid = int(row.get('persona_id', '').strip())
                    except Exception:
                        continue

                    emb_path_raw = row.get('embedding_path', '')
                    emb_path = _resolve_embedding_path(emb_path_raw, rec_file, db_root)

                    # 兼容旧格式记录：无 embedding_path 时，尝试根据 image_path 推断
                    if (not emb_path_raw) or (not os.path.exists(emb_path)):
                        image_path = row.get('image_path', '').strip()
                        if image_path:
                            image_name = os.path.splitext(os.path.basename(image_path))[0]
                            inferred_emb = os.path.join(os.path.dirname(rec_file), 'embeddings', f'{image_name}.npy')
                            if os.path.exists(inferred_emb):
                                emb_path = inferred_emb

                    if not emb_path or not os.path.exists(emb_path):
                        continue

                    try:
                        emb = np.load(emb_path)
                        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                    except Exception:
                        continue

                    if emb.size == 0:
                        continue

                    pid_to_embeddings.setdefault(pid, []).append(emb)
        except Exception as e:
            logger.warning(f"读取历史记录失败: {rec_file}, 错误: {e}")

    # 回退策略：如果CSV无法恢复，直接扫描embeddings目录并从文件名提取persona_id
    if not pid_to_embeddings:
        pid_pattern = re.compile(r'face_(\d+)')
        for root, _, files in os.walk(db_root):
            if os.path.basename(root) != 'embeddings':
                continue
            for fn in files:
                if not fn.lower().endswith('.npy'):
                    continue
                m = pid_pattern.search(fn)
                if not m:
                    continue
                pid = int(m.group(1))
                emb_path = os.path.join(root, fn)
                try:
                    emb = np.load(emb_path)
                    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                except Exception:
                    continue
                if emb.size == 0:
                    continue
                pid_to_embeddings.setdefault(pid, []).append(emb)

    if not pid_to_embeddings:
        return 0

    loaded = 0
    max_pid = 0
    for pid, emb_list in sorted(pid_to_embeddings.items()):
        if not emb_list:
            continue

        # 同一 persona 可能有多个样本，使用均值向量作为代表
        try:
            proto = np.mean(np.stack(emb_list, axis=0), axis=0)
            proto = proto / (np.linalg.norm(proto) + 1e-12)
        except Exception:
            continue

        deduper.embeddings.append(proto.astype(np.float32))
        deduper.ids.append(pid)
        deduper.metadata.append({'loaded_from_db': True})
        loaded += 1
        max_pid = max(max_pid, pid)

    # 继续从最大ID往后分配，避免ID冲突
    deduper._next_id = max(deduper._next_id, max_pid + 1)
    return loaded


class FrontalFaceExtractor:
    """
    正脸提取器
    
    核心策略：
    1. 只保存正脸（姿态角度在阈值内）
    2. 使用 track 信息关联同一人的不同角度
    3. 每个 track 只保存质量最好的正脸
    """
    
    def __init__(self, args):
        self.args = args
        self.det = Detector(backend=args.detector, device='cuda' if args.cuda else 'cpu')
        self.deduper = Deduper(metric=args.metric, threshold=args.threshold)
        self.embedder = SimpleEmbedder()
        self.pose_estimator = HeadPoseEstimator()
        
        # 质量检验器（用于保存后的人脸图像质量检查）
        # 参数从 config 中读取，便于统一调整
        try:
            self.quality_validator = FaceSaveQualityValidator(**cfg.FACE_SAVE_VALIDATOR)
        except Exception:
            # 回退到默认值以防配置项缺失
            self.quality_validator = FaceSaveQualityValidator()
        
        # 从detector中获取人脸有效性检查器（ONNX模式下使用）
        self.face_validity_checker = self.det.face_validator if self.det.face_validator is not None else None

        # 轻量检测驱动追踪器（可选）
        self.detection_tracker = DetectionTracker() if getattr(args, 'use_detection_tracker', False) else None
        
        # 跟踪历史（用于检测连续重复人脸）
        self.track_history = TrackHistory(max_age=30)
        
        # YOLO tracking model
        self.model_track = None
        self.device_str = 'cuda' if args.cuda else 'cpu'
        self.use_half = bool(args.cuda)
        
        # 仍然尝试初始化 YOLO 跟踪模型以启用 ByteTrack（可与 InsightFace 配合）
        if args.use_tracks:
            try:
                from ultralytics import YOLO
                import os
                from pathlib import Path
                
                model_for_track = None
                
                # 优先级 1: 检查是否可以使用 det.model 进行tracking
                if (hasattr(self.det, 'model') and self.det.model is not None and 
                    not isinstance(self.det.model, str)):
                    # det.model 是 YOLO 对象，检查是否是 .pt 格式
                    try:
                        # 尝试直接使用，YOLO 对象原生支持 tracking
                        model_for_track = self.det.model
                        logger.info(f"使用检测器的 YOLO 模型进行 ByteTrack 跟踪")
                    except Exception as e:
                        logger.debug(f"检测器模型不支持 tracking: {e}")
                        model_for_track = None
                
                # 优先级 2: 逐个查询可用的 .pt 模型
                if model_for_track is None:
                    yolo_pt_models = [
                        'models/yolo/yolov11n-face.pt',
                        'models/yolo/yolov11s-face.pt',
                        'models/yolo/yolo26n-face.pt',
                        'models/yolo/yolov10n-face.pt',
                        'models/yolo/yolov11m-face.pt',
                        'models/yolo/yolov11l-face.pt',
                        './yolov11n-face.pt',
                        './yolo26n-face.pt',
                        'yolov11n-face.pt',
                    ]
                    
                    for model_path in yolo_pt_models:
                        if os.path.exists(model_path):
                            try:
                                logger.info(f"找到 .pt 模型: {model_path}，加载用于 tracking...")
                                model_for_track = YOLO(model_path)
                                logger.info(f"✅ 成功加载 tracking 模型: {model_path}")
                                break
                            except Exception as e:
                                logger.debug(f"加载 {model_path} 失败: {e}")
                                continue
                
                # 优先级 3: 从 YOLO 目录自动查询
                if model_for_track is None:
                    yolo_dir = Path('models/yolo')
                    if yolo_dir.exists():
                        pt_files = sorted(yolo_dir.glob('*.pt'))
                        for pt_file in pt_files:
                            try:
                                logger.info(f"尝试加载: {pt_file}")
                                model_for_track = YOLO(str(pt_file))
                                logger.info(f"✅ 成功从目录加载 tracking 模型: {pt_file}")
                                break
                            except Exception as e:
                                logger.debug(f"无法加载 {pt_file}: {e}")
                                continue
                
                if model_for_track is not None:
                    self.model_track = model_for_track
                    logger.info("✅ ByteTrack 跟踪已启用")
                else:
                    logger.warning(
                        "⚠️  未找到支持 tracking 的 .pt 模型，将禁用 ByteTrack 跟踪\n"
                        "   ~ YOLO 会在首次运行时自动下载模型到 models/yolo/\n"
                        "   ~ 或手动下载: wget -O models/yolo/yolo26n-face.pt \\\n"
                        "     https://github.com/akanametov/yolo-face/releases/download/v1.0.0/yolo26n-face.pt"
                    )
                    self.model_track = None
            except Exception as e:
                logger.error(f"❌ 未能初始化 YOLO 跟踪器: {e}")
                logger.warning("ByteTrack 跟踪已禁用")
                self.model_track = None
    
    def process_frame_with_tracking(self, frame: np.ndarray, frame_count: int, 
                                    timestamp: float, time_str: str,
                                    output_dir: str, rec_path: str,
                                    track_db: Dict[int, FaceRecord],
                                    previous_active: set) -> Tuple[set, int]:
        """
        使用跟踪处理单帧
        
        步骤流程：
        1. ByteTrack.track() - 获取bbox和检测置信度
        2. 对frame级别进行一次detect - 获取kps（关键点）
        3. 在原始frame坐标系中进行姿态估计
        4. 利用过滤后的结果进行后续处理
        
        Returns:
            (current_active, saved_count)
        """
        saved_count = 0
        current_active = set()
        H, W = frame.shape[:2]
        
        # 如果既没有 ByteTrack 模型，也没有轻量 DetectionTracker，则直接返回
        if getattr(self, 'detection_tracker', None) is None and self.model_track is None:
            return current_active, saved_count

        # 标记：是否已经执行过 frame-level detect（用于避免重复调用 detect）
        detection_done = False
        
        # ============= 步骤 1: 人脸跟踪 (track) - 获得bbox和id =============
        # 如果配置为使用轻量 DetectionTracker，则先进行 frame-level detect，
        # 然后调用 detection_tracker.update() 得到 track id 列表（center-distance + size 匹配）
        xyxy = np.zeros((0, 4), dtype=np.int32)
        confs = np.array([])
        ids = np.array([], dtype=int)

        if getattr(self, 'detection_tracker', None) is not None:
            try:
                frame_detections = self.det.detect(frame, conf_threshold=self.args.conf)
            except Exception as e:
                frame_detections = []
                logger.debug(f"frame级别检测失败(DetectionTracker模式): {e}")

            boxes_xyxy = []
            boxes_confs = []
            for det in frame_detections:
                bbox = det.get('bbox')
                if not bbox:
                    continue
                boxes_xyxy.append(tuple(map(int, bbox)))
                boxes_confs.append(float(det.get('confidence', det.get('conf', self.args.conf))))

            if len(boxes_xyxy) == 0:
                return current_active, saved_count

            # 调用轻量 tracker
            try:
                assigned = self.detection_tracker.update(
                    [{'bbox': b, 'confidence': c} for b, c in zip(boxes_xyxy, boxes_confs)],
                    frame_count
                )
            except Exception as e:
                logger.warning(f"DetectionTracker 更新失败: {e}")
                return current_active, saved_count

            xyxy = np.array(boxes_xyxy, dtype=np.int32)
            confs = np.array(boxes_confs, dtype=float)
            ids = np.array([int(t) if t is not None else -1 for t in assigned], dtype=int)
            detection_done = True

        # 如果没有使用轻量 tracker，则回退到 ByteTrack (Ultralytics) 的 track()
        if not detection_done:
            try:
                # 暂时关闭 stderr (FD 2) 以抑制 ONNX Runtime C++ 日志
                old_stderr = sys.stderr.fileno() if hasattr(sys.stderr, 'fileno') else None
                if old_stderr is not None:
                    old_fd = os.dup(old_stderr)
                    devnull_fd = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull_fd, old_stderr)
                
                try:
                    track_results = self.model_track.track(
                        frame, persist=True, conf=self.args.conf,
                        tracker="bytetrack.yaml", verbose=False,
                        half=self.use_half, device=self.device_str, task='detect'
                    )
                finally:
                    # 恢复 stderr
                    if old_stderr is not None:
                        os.dup2(old_fd, old_stderr)
                        os.close(old_fd)
                        os.close(devnull_fd)
            except Exception as e:
                logger.warning(f"跟踪失败: {e}")
                return current_active, saved_count
            
            if not track_results or len(track_results) == 0:
                return current_active, saved_count
            
            boxes_track = getattr(track_results[0], 'boxes', None)
            if boxes_track is None or len(boxes_track) == 0:
                return current_active, saved_count

            xyxy = boxes_track.xyxy.cpu().numpy()
            confs = boxes_track.conf.cpu().numpy()
            ids = boxes_track.id.cpu().numpy().astype(int) if boxes_track.id is not None else np.array([-1] * len(xyxy))
        
        # ============= 步骤 2: 人脸检测 (detect) - 获得kps（关键点）=============
        # 在整个frame上进行一次detect，以获得关键点信息
        # 如果已在 DetectionTracker 分支中完成 detect，则跳过重复调用
        if not detection_done:
            frame_detections = []
            try:
                frame_detections = self.det.detect(frame, conf_threshold=self.args.conf)
            except Exception as e:
                logger.debug(f"frame级别检测失败: {e}")
        
        # 构建一个kps查找表：根据bbox匹配找到对应的kps
        bbox_to_kps = {}
        if frame_detections:
            for det in frame_detections:
                if det and det.get('kps') is not None:
                    bbox = det.get('bbox')
                    if bbox:
                        # 以bbox作为key（四舍五入到整数）
                        bbox_key = tuple(map(int, bbox))
                        bbox_to_kps[bbox_key] = det.get('kps')
        
        # ============= 帧级去重：收集本帧的所有人脸embedding =============
        # 同一帧中可能出现多个人脸，如果相似度高应该去重
        frame_embeddings = []  # [(tid, bbox, kps, emb, quality_result), ...]
        
        # ============= 处理每个tracked box =============
        
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[j])
            tid = int(ids[j])
            conf = float(confs[j])
            
            if tid == -1:
                continue
            
            current_active.add(tid)
            
            # 裁剪人脸
            cx1, cy1 = max(0, min(x1, W-1)), max(0, min(y1, H-1))
            cx2, cy2 = max(0, min(x2, W-1)), max(0, min(y2, H-1))
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            
            face_crop = frame[cy1:cy2, cx1:cx2]
            
            # ============= 步骤 3 & 4: 提取关键点和评估质量 =============
            # 策略：优先使用frame-level detect的kps（复用步骤2的输出）
            kps = None
            det_embedding = None
            
            # 尝试从frame-level detection的结果中找到匹配的kps
            bbox_key = (x1, y1, x2, y2)
            if bbox_key in bbox_to_kps:
                kps = bbox_to_kps[bbox_key]
            else:
                # 如果没有精确匹配，尝试近似匹配（允许一些像素偏差）
                for stored_bbox, stored_kps in bbox_to_kps.items():
                    if (abs(stored_bbox[0]-x1) < 5 and abs(stored_bbox[1]-y1) < 5 and
                        abs(stored_bbox[2]-x2) < 5 and abs(stored_bbox[3]-y2) < 5):
                        kps = stored_kps
                        break
            
            # 如果仍然没有kps，尝试在face_crop上进行第二次检测（备用方案）
            if kps is None:
                try:
                    found = self.det.detect(face_crop, conf_threshold=self.args.conf)
                    if found and found[0]:
                        det_info = found[0]
                        kps = det_info.get('kps')
                        det_embedding = det_info.get('embedding')
                except Exception:
                    pass
            
            # 评估人脸质量（使用完整的检测信息和实际图像）
            quality_result = evaluate_face_quality(
                {'bbox': (x1, y1, x2, y2), 'kps': kps, 'confidence': conf},
                image_shape=frame.shape[:2],
                face_image=face_crop,  # 传入实际人脸裁剪用于物理质量检查
                yaw_threshold=self.args.yaw_threshold,
                pitch_threshold=self.args.pitch_threshold,
                roll_threshold=self.args.roll_threshold,
                debug=True  # 启用调试日志以输出筛选详情
            )
            
            # =============== 额外验证：如果没有关键点（ONNX模式），进行人脸有效性检查 ===============
            # 防止鼠标、手机等误检
            if kps is None and self.face_validity_checker is not None:
                is_valid, val_scores = self.face_validity_checker.is_valid_face(frame, (x1, y1, x2, y2), conf)
                if not is_valid:
                    category, reason, _ = self.face_validity_checker.classify_detection(frame, (x1, y1, x2, y2), conf)
                    logger.debug(f"⚠️  检测到非人脸对象 (轨迹{tid}): {reason}")
                    logger.debug(f"   皮肤色比={val_scores.get('skin_ratio', 0):.2f}, " \
                                f"熵={val_scores.get('entropy', 0):.2f}, " \
                                f"边缘={val_scores.get('edge_density', 0):.2f}, " \
                                f"宽高比={val_scores.get('aspect_ratio', 0):.2f}")
                    continue  # 跳过这个检测
            
            # 对齐人脸（使用之前获取的kps，或从检测中获取）
            aligned = align_face(face_crop, kps)
            
            # ============= 步骤 7: 提取 Embedding =============
            # 优先使用frame-level detection中的embedding，否则从原始人脸裁剪中提取
            emb = None
            if det_embedding is not None:
                emb = np.asarray(det_embedding, dtype=np.float32)
                # 归一化
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            else:
                # 备用方案：使用原始 face_crop（而非aligned）的 HSV 直方图
                emb = SimpleEmbedder.get_embedding_from_detection({}, face_crop)
            
            # 获取或创建 track 记录
            rec = track_db.get(tid)
            if rec is None:
                rec = FaceRecord(track_id=tid)
                track_db[tid] = rec
            
            # 更新 track 记录
            if emb is not None:
                rec.embeddings.append(emb)
            
            # 只处理正脸
            if quality_result.is_frontal:
                rec.frontal_count += 1
                
                # 更新最佳正脸
                if quality_result.quality_score > rec.best_quality_score:
                    rec.best_quality_score = quality_result.quality_score
                    rec.best_pose_score = quality_result.pose_score
                    rec.best_image = aligned.copy() if aligned is not None else face_crop.copy()
                    rec.best_yaw = quality_result.yaw
                    rec.best_pitch = quality_result.pitch
                    rec.best_roll = quality_result.roll
            
            # =============== 本帧去重：收集正脸embedding =================
            # 同一帧中如果有多个相似的人脸，进行实时去重
            if quality_result.is_frontal and emb is not None:
                frame_embeddings.append({
                    'tid': tid,
                    'emb': emb,
                    'quality_score': quality_result.quality_score,
                    'aligned': aligned.copy() if aligned is not None else face_crop.copy(),
                    'rec': rec
                })
            
            # 保存策略: first - 第一个正脸立即保存
            if self.args.save_strategy == 'first' and not rec.first_saved:
                if quality_result.is_frontal and emb is not None:
                    logger.debug(f"📹 正在处理轨迹ID={tid} | 帧={frame_count} | 质量分数={quality_result.quality_score:.4f}")
                    
                    # ================ 连续重复检查：避免保存同一person的重复帧 ================
                    # 使用 track_history 检查是否应该保存（连续出现则跳过）
                    should_save_this_frame = self.track_history.update(tid, frame_count, quality_result.quality_score)
                    if not should_save_this_frame:
                        logger.info(f"⏭️  跳过连续重复帧: track{tid} 在frame{frame_count} 连续出现，质量不如最好帧")
                        continue  # 跳过这一帧的保存
                    
                    pid = self.deduper.find_match(emb, debug=True)
                    if pid is None:
                        pid = self.deduper.add(emb, {
                            'yaw': quality_result.yaw,
                            'pitch': quality_result.pitch,
                            'roll': quality_result.roll
                        })
                        
                        # 直接保存原始人脸裁剪和 embedding（质量筛选已完成）
                        os.makedirs(os.path.join(output_dir, 'embeddings'), exist_ok=True)
                        os.makedirs(os.path.join(output_dir, 'debug_frames'), exist_ok=True)  # 调试用
                        img_path = os.path.join(output_dir, f"face_{pid:05d}_track{tid}_f{frame_count}.jpg")
                        emb_path = os.path.join(output_dir, 'embeddings', f"face_{pid:05d}_track{tid}_f{frame_count}.npy")
                        frame_path = os.path.join(output_dir, 'debug_frames', f"frame_{pid:05d}_track{tid}_f{frame_count}.jpg")  # 保存原始帧
                        
                        try:
                            # 保存原始人脸裁剪（已通过质量筛选）- 不使用对齐
                            face_saved = cv2.imwrite(img_path, face_crop)
                            # 保存原始帧截图（用于对比分析）
                            frame_saved = cv2.imwrite(frame_path, frame)
                            
                            if face_saved and frame_saved:
                                # 保存 embedding（应该来自 InsightFace，质量更高）
                                if emb is not None and emb.size > 0:
                                    np.save(emb_path, emb.astype(np.float32))
                                    saved_count += 1
                                    rec.first_saved = True
                                    
                                    # 保存质量检测详细信息到文本文件
                                    debug_info_path = os.path.join(output_dir, 'debug_frames', f"frame_{pid:05d}_track{tid}_f{frame_count}_info.txt")
                                    with open(debug_info_path, 'w', encoding='utf-8') as f:
                                        f.write(f"人脸ID: {pid:05d}\n")
                                        f.write(f"轨迹ID: {tid}\n")
                                        f.write(f"帧号: {frame_count}\n")
                                        f.write(f"时间: {time_str} ({timestamp:.2f}s)\n")
                                        f.write(f"检测置信度: {conf:.4f}\n")
                                        f.write(f"质量分数: {quality_result.quality_score:.4f}\n")
                                        f.write(f"姿态分数: {quality_result.pose_score:.4f}\n")
                                        f.write(f"头部姿态: Y={quality_result.yaw:.1f}°, P={quality_result.pitch:.1f}°, R={quality_result.roll:.1f}°\n")
                                        f.write(f"是否高质量: {quality_result.is_high_quality}\n")
                                        f.write(f"是否正脸: {quality_result.is_frontal}\n")
                                        if quality_result.reasons:
                                            f.write(f"质量问题:\n")
                                            for reason in quality_result.reasons:
                                                f.write(f"  - {reason}\n")
                                        f.write(f"人脸坐标: ({x1}, {y1}, {x2}, {y2})\n")
                                        f.write(f"人脸尺寸: {x2-x1}x{y2-y1}\n")
                                    
                                    logger.info(f"✅ 保存首帧正脸: pid={pid:05d} | track={tid} | frame={frame_count} | 质量={quality_result.quality_score:.4f} | "
                                              f"姿态(Y={quality_result.yaw:.1f}°, P={quality_result.pitch:.1f}°, R={quality_result.roll:.1f}°) | emb_dim={emb.size}")
                                    with open(rec_path, 'a', encoding='utf-8') as f:
                                        f.write(f"{pid},{tid},{time_str},{timestamp:.2f},{conf:.4f},{img_path},{emb_path},{frame_count}\n")
                                else:
                                    logger.warning(f"⚠️  embedding 无效: pid={pid:05d} | track={tid}")
                                    try:
                                        os.remove(img_path)
                                        os.remove(frame_path)
                                    except:
                                        pass
                            else:
                                logger.error(f"❌ 保存图像失败: img={face_saved}, frame={frame_saved}")
                        except Exception as e:
                            logger.error(f"❌ 保存人脸时出错: {e}")
        
        # =============== 帧级去重处理：对本帧的多个人脸进行去重 ===============
        if len(frame_embeddings) > 1:
            logger.debug(f"🔍 本帧检测到 {len(frame_embeddings)} 个人脸，执行帧级去重...")
            
            # 从高中到低排序（优先保留质量最好的）
            frame_embeddings.sort(key=lambda x: x['quality_score'], reverse=True)
            
            # 建立重复人脸映射：{重复tid: 主tid}
            duplicate_mapping = {}
            kept_tids = set()
            
            for i, item_i in enumerate(frame_embeddings):
                if item_i['tid'] in duplicate_mapping:
                    continue  # 已经被标记为重复
                
                kept_tids.add(item_i['tid'])
                emb_i = item_i['emb']
                
                # 与后续人脸比较
                for j in range(i + 1, len(frame_embeddings)):
                    item_j = frame_embeddings[j]
                    if item_j['tid'] in duplicate_mapping:
                        continue
                    
                    emb_j = item_j['emb']
                    
                    # 计算相似度
                    sim = float(np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-12))
                    
                    # 如果相似度高于阈值，标记为重复
                    if sim >= self.args.threshold:
                        duplicate_mapping[item_j['tid']] = item_i['tid']
                        logger.info(f"🔗 帧级去重: track{item_j['tid']} (sim={sim:.4f}) → track{item_i['tid']} (质量更优)")
                        kept_tids.discard(item_j['tid'])
        
        return current_active, saved_count
    
    def process_ended_tracks(self, ended_tracks: set, track_db: Dict[int, FaceRecord],
                            output_dir: str, rec_path: str, time_str: str, 
                            timestamp: float, frame_count: int) -> int:
        """
        处理结束的 track（best_end 策略）
        
        只保存有正脸的 track，且保存质量最好的正脸
        """
        saved_count = 0
        
        for tid in list(ended_tracks):
            rec = track_db.get(tid)
            if not rec:
                continue
            
            # 检查是否有正脸
            if rec.frontal_count == 0 or rec.best_image is None:
                track_db.pop(tid, None)
                continue
            
            # 计算聚合 embedding
            if len(rec.embeddings) == 0:
                track_db.pop(tid, None)
                continue
            
            agg_emb = np.mean(np.stack(rec.embeddings, axis=0), axis=0)
            
            # 去重匹配
            logger.debug(f"📹 处理轨迹ID={tid} | 共{len(rec.embeddings)}帧 | 质量={rec.best_quality_score:.4f}")
            pid = self.deduper.find_match(agg_emb, debug=False)
            if pid is None:
                pid = self.deduper.add(agg_emb, {
                    'yaw': rec.best_yaw,
                    'pitch': rec.best_pitch,
                    'roll': rec.best_roll
                })
            
            # 保存最佳正脸
            img_path = os.path.join(output_dir, f"face_{pid:05d}_track{tid}_FRONTAL.jpg")
            emb_path = os.path.join(output_dir, 'embeddings', f"face_{pid:05d}_track{tid}_FRONTAL.npy")
            
            # 保存最佳图像并验证质量
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            if cv2.imwrite(img_path, rec.best_image):
                # 简单的质量验证（对于已对齐的图像）
                is_valid, reason = self.quality_validator.validate(rec.best_image, img_path)
                if is_valid and save_embedding(agg_emb, emb_path):
                    saved_count += 1
                    with open(rec_path, 'a', encoding='utf-8') as f:
                        f.write(f"{pid},{tid},{time_str},{timestamp:.2f},{rec.best_quality_score:.4f},{img_path},{emb_path},{frame_count}\n")
                else:
                    # 质量检验失败，删除图像
                    logger.warning(f"⚠️  轨迹{tid}的最佳人脸质量检验失败: {reason}")
                    try:
                        os.remove(img_path)
                    except:
                        pass
            
            track_db.pop(tid, None)
        
        return saved_count


def process_video(video_path: str, output_dir: str, args):
    """处理单个视频"""
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = FrontalFaceExtractor(args)

    # 加载历史 embedding 库，支持和已有人脸ID进行匹配
    if getattr(args, 'reuse_embedding_db', True):
        loaded = load_existing_embeddings_to_deduper(extractor.deduper, args.embedding_db_dir)
        if loaded > 0:
            logger.info(f"已加载历史人脸库: {loaded} 个persona (db: {args.embedding_db_dir})")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_interval = args.sample_interval if args.sample_interval and args.sample_interval > 0 else max(int(round(fps)), 1)
    
    # 预览输出
    preview_out = os.path.join(output_dir, 'preview.mp4')
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        return 0
    H, W = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vw = cv2.VideoWriter(preview_out, cv2.VideoWriter_fourcc(*'mp4v'), max(fps / sample_interval, 1), (W, H))
    
    # 记录文件
    rec_path = os.path.join(output_dir, 'face_records_frontal.txt')
    if not os.path.exists(rec_path) or os.path.getsize(rec_path) == 0:
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write('persona_id,track_id,time,timestamp,quality_score,image_path,embedding_path,frame_count\n')
    
    frame_count = 0
    saved_count = 0
    track_db: Dict[int, FaceRecord] = {}
    previous_active = set()
    
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=f"FrontalFace: {Path(video_path).name}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if total_frames > 0:
                pbar.update(1)
            
            if frame_count % sample_interval != 0:
                frame_count += 1
                continue
            
            timestamp = frame_count / fps
            time_str = time.strftime('%H:%M:%S', time.gmtime(int(timestamp)))
            
            if args.use_tracks and extractor.model_track is not None:
                # 使用跟踪模式
                current_active, frame_saved = extractor.process_frame_with_tracking(
                    frame, frame_count, timestamp, time_str,
                    output_dir, rec_path, track_db, previous_active
                )
                saved_count += frame_saved
                
                # 处理结束的 track
                if args.save_strategy == 'best_end':
                    ended = previous_active - current_active
                    ended_saved = extractor.process_ended_tracks(
                        ended, track_db, output_dir, rec_path,
                        time_str, timestamp, frame_count
                    )
                    saved_count += ended_saved
                
                previous_active = current_active
            else:
                # 不使用跟踪模式：逐帧检测
                dets = extractor.det.detect(frame, conf_threshold=args.conf)
                for d in dets:
                    # 先裁剪人脸区域，以便进行物理质量检查
                    x1, y1, x2, y2 = d['bbox']
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # 评估质量（包括物理图像质量）
                    quality_result = evaluate_face_quality(
                        d, image_shape=frame.shape[:2],
                        face_image=face_crop,  # 传入实际人脸裁剪
                        yaw_threshold=args.yaw_threshold,
                        pitch_threshold=args.pitch_threshold,
                        roll_threshold=args.roll_threshold,
                        debug=True  # 启用调试日志以输出筛选详情
                    )
                    
                    # 只处理正脸
                    if not quality_result.is_frontal:
                        continue
                    
                    aligned = align_face(face_crop, d.get('kps'))
                    emb = SimpleEmbedder.get_embedding_from_detection(d, aligned)
                    
                    if emb is None:
                        continue
                    
                    logger.debug(f"📹 正在处理人脸检测 | 帧={frame_count} | 质量分数={quality_result.quality_score:.4f}")
                    pid = extractor.deduper.find_match(emb, debug=False)
                    if pid is None:
                        pid = extractor.deduper.add(emb, {
                            'yaw': quality_result.yaw,
                            'pitch': quality_result.pitch,
                            'roll': quality_result.roll
                        })
                        img_path = os.path.join(output_dir, f"face_{pid:05d}_f{frame_count}.jpg")
                        emb_path = os.path.join(output_dir, 'embeddings', f"face_{pid:05d}_f{frame_count}.npy")
                        
                        if save_image(aligned, img_path) and save_embedding(emb, emb_path):
                            saved_count += 1
                            with open(rec_path, 'a', encoding='utf-8') as f:
                                f.write(f"{pid},-1,{time_str},{timestamp:.2f},{quality_result.quality_score:.4f},{img_path},{emb_path},{frame_count}\n")
            
            vw.write(frame)
            frame_count += 1
    
    finally:
        pbar.close()
        cap.release()
        vw.release()
        
        # 处理剩余的 track
        if args.save_strategy == 'best_end' and track_db:
            logger.info(f"处理剩余 {len(track_db)} 个 track...")
            for tid, rec in list(track_db.items()):
                if rec.frontal_count > 0 and rec.best_image is not None and len(rec.embeddings) > 0:
                    agg_emb = np.mean(np.stack(rec.embeddings, axis=0), axis=0)
                    logger.debug(f"📹 处理最后的轨迹ID={tid} | 共{len(rec.embeddings)}帧 | 质量={rec.best_quality_score:.4f}")
                    pid = extractor.deduper.find_match(agg_emb, debug=False)
                    if pid is None:
                        pid = extractor.deduper.add(agg_emb)
                        img_path = os.path.join(output_dir, f"face_{pid:05d}_track{tid}_FINAL.jpg")
                        
                        # 保存并验证质量
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        if cv2.imwrite(img_path, rec.best_image):
                            is_valid, reason = extractor.quality_validator.validate(rec.best_image, img_path)
                            if is_valid:
                                saved_count += 1
                            else:
                                logger.warning(f"⚠️  轨迹{tid}的最终人脸质量检验失败: {reason}")
                                try:
                                    os.remove(img_path)
                                except:
                                    pass
    
    logger.info(f"完成: {saved_count} 张正脸已保存 (输出: {output_dir})")
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description='高质量正脸提取流水线（改进版：姿态过滤 + 去重）'
    )
    parser.add_argument('input', type=str, help='视频文件或包含视频的目录')
    parser.add_argument('--output-dir', '-o', type=str, default=cfg.DEFAULT_OUTPUT_DIR, help='输出目录')
    parser.add_argument('--conf', type=float, default=cfg.DEFAULT_CONFIDENCE_THRESHOLD, help='检测置信度阈值')
    parser.add_argument('--sample-interval', type=int, default=cfg.DEFAULT_SAMPLE_INTERVAL, help='采样间隔（帧）')

    # 跟踪选项
    parser.add_argument('--use-tracks', dest='use_tracks', action='store_true', help='使用跟踪')
    parser.add_argument('--no-tracks', dest='use_tracks', action='store_false')
    parser.set_defaults(use_tracks=cfg.USE_TRACKING)
    parser.add_argument('--use-detection-tracker', dest='use_detection_tracker', action='store_true',
                       help='使用轻量级检测驱动的跟踪器(DetectionTracker)，以InsightFace检测结果驱动跟踪，优先于 ByteTrack')
    parser.add_argument('--no-detection-tracker', dest='use_detection_tracker', action='store_false',
                       help='禁用轻量级检测驱动的跟踪器（默认启用）')
    parser.set_defaults(use_detection_tracker=True)

    # 保存策略
    parser.add_argument('--save-strategy', type=str, choices=['first', 'best_end'], default=cfg.SAVE_STRATEGY,
                       help='保存策略: first(第一个正脸), best_end(track结束时保存最佳)')

    # 姿态阈值
    parser.add_argument('--yaw-threshold', type=float, default=cfg.YAW_THRESHOLD, help='Yaw角度阈值（度）')
    parser.add_argument('--pitch-threshold', type=float, default=cfg.PITCH_THRESHOLD, help='Pitch角度阈值（度）')
    parser.add_argument('--roll-threshold', type=float, default=cfg.ROLL_THRESHOLD, help='Roll角度阈值（度）')

    # 去重参数
    parser.add_argument('--threshold', type=float, default=cfg.DEDUP_THRESHOLD, help='去重相似度阈值')
    parser.add_argument('--metric', type=str, default=cfg.SIMILARITY_METRIC, choices=['cosine', 'euclidean'])

    # 其他
    parser.add_argument('--detector', type=str, default=cfg.DEFAULT_DETECTOR, choices=['auto', 'insightface', 'yolo'],
                       help='检测器: auto/insightface/yolo')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='使用GPU')
    parser.add_argument('--cpu', dest='cuda', action='store_false', help='强制使用CPU')
    parser.set_defaults(cuda=cfg.USE_CUDA)

    # 历史 embedding 库复用
    parser.add_argument('--reuse-embedding-db', dest='reuse_embedding_db', action='store_true',
                       help='加载历史embedding库并参与匹配（默认开启）')
    parser.add_argument('--no-reuse-embedding-db', dest='reuse_embedding_db', action='store_false',
                       help='不加载历史embedding库，仅在当前运行内匹配')
    parser.set_defaults(reuse_embedding_db=True)
    parser.add_argument('--embedding-db-dir', type=str, default=None,
                       help='历史embedding库目录（默认使用output-dir）')
    
    args = parser.parse_args()
    if args.embedding_db_dir is None:
        args.embedding_db_dir = args.output_dir
    
    # GPU 检测
    logger.info("=" * 60)
    logger.info("GPU 检测与配置:")
    gpu_available = detect_gpu_availability()
    
    # 如果指定使用GPU但GPU不可用，降级到CPU
    if args.cuda and not gpu_available:
        logger.warning("❌ 您请求使用GPU，但GPU不可用，已自动改为CPU模式")
        args.cuda = False
    
    device_str = "GPU (CUDA)" if args.cuda else "CPU"
    logger.info(f"✓ 处理设备: {device_str}")
    logger.info("=" * 60)
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("正脸提取配置:")
    logger.info(f"  姿态阈值: yaw={args.yaw_threshold}°, pitch={args.pitch_threshold}°, roll={args.roll_threshold}°")
    logger.info(f"  去重阈值: {args.threshold} ({args.metric})")
    logger.info(f"  保存策略: {args.save_strategy}")
    logger.info(f"  使用跟踪: {args.use_tracks}")
    logger.info("=" * 60)
    
    # 处理输入
    inp = args.input
    to_process = []
    exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    
    if os.path.isfile(inp):
        to_process = [inp]
    elif os.path.isdir(inp):
        for root, _, files in os.walk(inp):
            for fn in files:
                if fn.lower().endswith(exts):
                    to_process.append(os.path.join(root, fn))
    else:
        logger.error(f"输入路径不存在: {inp}")
        return
    
    for vid in to_process:
        video_name = Path(vid).stem
        per_out = os.path.join(args.output_dir, video_name)
        os.makedirs(per_out, exist_ok=True)
        process_video(vid, per_out, args)


if __name__ == '__main__':
    main()
