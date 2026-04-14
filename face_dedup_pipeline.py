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
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
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
from simple_face_validator import SimpleFaceValidator

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ReID 深度特征提取器（OSNet-x0.25 MSMT17）
# ============================================================================

class ReIDFeatureExtractor:
    """
    基于 OSNet-x0.25 (MSMT17 预训练) 的半身图片 ReID 特征提取器。
    输出 512 维 L2 归一化 embedding，比 HSV 直方图有更强的语义区分能力。
    """
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.model = None
        self.device = device
        self.input_h, self.input_w = cfg.REID_INPUT_SIZE
        self._loaded = False

        if model_path is None:
            model_path = cfg.REID_MODEL_PATH
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        self.model_path = model_path

    def _lazy_load(self):
        """延迟加载模型（首次推理时才加载 GPU 显存）。"""
        if self._loaded:
            return
        try:
            import torch
            if not os.path.exists(self.model_path):
                logger.warning(
                    f'⚠️ ReID 模型文件不存在: {self.model_path}，将回退到 HSV 直方图特征。'
                    f'请运行 python setup_project.py 下载模型。'
                )
                return

            if torch.cuda.is_available() and self.device != 'cpu':
                self.model = torch.jit.load(self.model_path, map_location='cuda').eval()
                self.device = 'cuda'
                logger.info(f'✅ ReID 模型已加载 (GPU): {self.model_path}')
            else:
                self.model = torch.jit.load(self.model_path, map_location='cpu').eval()
                self.device = 'cpu'
                logger.info(f'✅ ReID 模型已加载 (CPU): {self.model_path}')

            self._loaded = True
        except ImportError:
            logger.warning('⚠️ PyTorch 未安装，无法使用 ReID 特征，将回退到 HSV 直方图。')
        except Exception as e:
            logger.warning(f'⚠️ ReID 模型加载失败: {e}，将回退到 HSV 直方图。')

    def is_available(self) -> bool:
        """检查模型是否可用。"""
        self._lazy_load()
        return self.model is not None

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        从半身图片中提取 512 维 ReID embedding。

        Args:
            image: BGR 格式的半身图片 (np.ndarray)

        Returns:
            L2 归一化的 512 维 float32 向量，失败返回 None。
        """
        if image is None or image.size == 0:
            return None

        self._lazy_load()
        if self.model is None:
            return None

        try:
            import torch

            # 预处理：BGR -> RGB, resize, normalize
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            # ImageNet 标准化
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            # HWC -> NCHW
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

            if self.device == 'cuda':
                tensor = tensor.cuda()

            with torch.no_grad():
                feat = self.model(tensor)

            # L2 归一化
            feat = feat.cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm <= 0 or feat.size == 0:
                return None
            return feat / norm

        except Exception as e:
            logger.debug(f'ReID 特征提取失败: {e}')
            return None


# 全局 ReID 提取器实例（延迟加载）
_reid_extractor: Optional[ReIDFeatureExtractor] = None

# 全局 ArcFace 提取器实例（延迟加载）
_arcface_extractor: Optional['ArcFaceFeatureExtractor'] = None


def get_reid_extractor(device: str = 'cuda') -> Optional[ReIDFeatureExtractor]:
    """获取全局 ReID 特征提取器（单例，延迟加载）。"""
    global _reid_extractor
    if _reid_extractor is None:
        _reid_extractor = ReIDFeatureExtractor(device=device)
    return _reid_extractor


# ---------------------------------------------------------------------------
# ArcFace 人脸特征提取器（基于 insightface.app.FaceAnalysis）
# ---------------------------------------------------------------------------

class ArcFaceFeatureExtractor:
    """
    基于 InsightFace w600k_r50 (ArcFace) 的人脸特征提取器。
    输出 512 维 L2 归一化 embedding，通过人脸五官精确判断是否同一人。

    特点（与 OSNet ReID 对比）：
      - ArcFace 看人脸五官，同一人 0.85~0.95+，不同人 0.30~0.55
      - OSNet 看半身衣着，同一人 0.70~0.80，不同人 0.55~0.70（区分度差）
    """

    def __init__(self, model_root: str = None, model_name: str = None, device: str = 'cuda'):
        self.app = None
        self.device = device
        self.model_root = model_root or cfg.ARCFACE_MODEL_ROOT
        self.model_name = model_name or cfg.ARCFACE_MODEL_NAME
        self.det_size = getattr(cfg, 'ARCFACE_DET_SIZE', (640, 640))
        self._loaded = False

    def _lazy_load(self):
        """延迟加载 InsightFace 模型。"""
        if self._loaded:
            return
        try:
            from insightface.app import FaceAnalysis

            kwargs = {
                'name': self.model_name,
                'root': os.path.abspath(self.model_root),
            }
            self.app = FaceAnalysis(**kwargs)

            ctx_id = 0 if (self.device == 'cuda') else -1
            try:
                self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            except Exception:
                self.app.prepare(ctx_id=-1, det_size=self.det_size)

            self._loaded = True
            logger.info(
                f'✅ ArcFace 模型已加载 (model={self.model_name}, '
                f'device={"GPU" if ctx_id == 0 else "CPU"})'
            )
        except ImportError:
            logger.warning('⚠️ insightface 未安装，无法使用 ArcFace 特征。将回退到 ReID/HSV。')
        except Exception as e:
            logger.warning(f'⚠️ ArcFace 模型加载失败: {e}。将回退到 ReID/HSV。')

    def is_available(self) -> bool:
        """检查模型是否可用。"""
        self._lazy_load()
        return self.app is not None

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        从图片中检测人脸并提取 ArcFace embedding。
        如果图片中有多张人脸，返回面积最大的人脸的 embedding。
        """
        if image is None or image.size == 0:
            return None

        self._lazy_load()
        if self.app is None:
            return None

        try:
            faces = self.app.get(image)
            if not faces:
                return None

            # 选择面积最大的人脸
            bbox_areas = [
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                for f in faces
            ]
            best_idx = int(np.argmax(bbox_areas))
            face = faces[best_idx]

            emb = face.embedding.astype(np.float32)
            if emb is None or emb.size == 0:
                return None

            # 确保 L2 归一化
            norm = np.linalg.norm(emb)
            if norm <= 0:
                return None
            return emb / norm

        except Exception as e:
            logger.debug(f'ArcFace 特征提取失败: {e}')
            return None


def get_arcface_extractor(device: str = 'cuda') -> Optional[ArcFaceFeatureExtractor]:
    """获取全局 ArcFace 特征提取器（单例，延迟加载）。"""
    global _arcface_extractor
    if _arcface_extractor is None:
        _arcface_extractor = ArcFaceFeatureExtractor(device=device)
    return _arcface_extractor


def extract_half_body_embedding(
    half_body_crop: Optional[np.ndarray],
    embedding_mode: str = 'half_body',
    use_reid: bool = True,
    hist_bins: int = 8,
    reid_extractor: Optional[ReIDFeatureExtractor] = None,
    cluster_feature_mode: str = 'reid',
    arcface_extractor: Optional[ArcFaceFeatureExtractor] = None,
) -> Optional[np.ndarray]:
    """
    统一的半身 embedding 提取入口。

    cluster_feature_mode 决定使用哪种特征：
      - 'arcface': 使用 ArcFace 人脸特征（推荐用于聚类，区分度最高）
      - 'reid'   : 使用 OSNet ReID 行人特征（看半身衣着）
      - 'hsv'    : 使用 HSV 颜色直方图（传统方法）
    当请求的模式不可用时，自动回退：arcface → reid → hsv
    """
    if half_body_crop is None or half_body_crop.size == 0:
        return None

    mode = (embedding_mode or 'half_body').strip().lower()
    if mode != 'half_body':
        return None  # face 模式不在此处理

    cluster_feature = (cluster_feature_mode or 'reid').strip().lower()

    # 优先 1: ArcFace 人脸特征
    if cluster_feature == 'arcface':
        if arcface_extractor is None:
            arcface_extractor = get_arcface_extractor()
        if arcface_extractor is not None and arcface_extractor.is_available():
            emb = arcface_extractor.extract(half_body_crop)
            if emb is not None:
                return emb
            logger.debug('ArcFace 提取返回空（未检测到人脸），回退到 ReID/HSV')
        # ArcFace 失败，回退到 reid
        cluster_feature = 'reid'

    # 优先 2: ReID 深度特征
    if cluster_feature == 'reid' and use_reid:
        if reid_extractor is None:
            reid_extractor = get_reid_extractor()
        if reid_extractor is not None and reid_extractor.is_available():
            emb = reid_extractor.extract(half_body_crop)
            if emb is not None:
                return emb
            logger.debug('ReID 提取返回空，回退到 HSV 直方图')

    # 回退: HSV 直方图
    return extract_apparel_context_hist(half_body_crop, hist_bins=hist_bins)


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


def extract_half_body_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    top_expand: float,
    bottom_expand: float,
    side_expand: float,
) -> Optional[np.ndarray]:
    """基于人脸框向下扩展，截取包含衣物的半身区域。"""
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)

    ex1 = max(0, int(round(x1 - fw * side_expand)))
    ex2 = min(w, int(round(x2 + fw * side_expand)))
    ey1 = max(0, int(round(y1 - fh * top_expand)))
    ey2 = min(h, int(round(y2 + fh * bottom_expand)))

    if ex2 <= ex1 or ey2 <= ey1:
        return None

    return frame[ey1:ey2, ex1:ex2]


def extract_apparel_context_hist(half_body_crop: Optional[np.ndarray], hist_bins: int = 8) -> Optional[np.ndarray]:
    """从半身图下半部分提取HSV直方图，作为衣物上下文特征。"""
    if half_body_crop is None or half_body_crop.size == 0:
        return None

    try:
        hh, ww = half_body_crop.shape[:2]
        if hh < 10 or ww < 10:
            return None

        # 重点取下半部分，减少脸部对衣物特征的干扰
        start_y = int(hh * 0.45)
        apparel_region = half_body_crop[start_y:, :]
        if apparel_region.size == 0:
            return None

        hsv = cv2.cvtColor(apparel_region, cv2.COLOR_BGR2HSV)
        bins = max(4, int(hist_bins))
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm <= 0 or hist.size == 0:
            return None
        return hist / norm
    except Exception:
        return None


def build_context_match_metadata(args, context_hist: Optional[np.ndarray]) -> Dict:
    """构建传给 Deduper.find_match 的上下文元信息。"""
    if (not getattr(args, 'use_context_reid', False)) or context_hist is None:
        return {}

    return {
        'context_hist': context_hist,
        'face_weight': float(getattr(args, 'context_face_weight', cfg.CONTEXT_FACE_WEIGHT)),
        'context_weight': float(getattr(args, 'context_apparel_weight', cfg.CONTEXT_APPAREL_WEIGHT)),
    }


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


def _is_embedding_path_for_mode(emb_path: str, embedding_mode: str) -> bool:
    """根据文件名判断 embedding 是否属于指定模式。"""
    mode = (embedding_mode or 'half_body').strip().lower()
    name = os.path.basename(emb_path or '').lower()
    if mode == 'half_body':
        return '_body' in name
    if mode == 'face':
        return '_body' not in name
    return True


def _resolve_image_path(image_path: str, rec_file_path: str, db_root: str) -> str:
    """将记录中的 image_path 解析为本地绝对路径。"""
    if not image_path:
        return ""

    p = image_path.strip()
    if os.path.isabs(p):
        return p

    p_clean = p[2:] if p.startswith('./') else p
    candidate_from_cwd = os.path.abspath(p)
    if os.path.exists(candidate_from_cwd):
        return candidate_from_cwd

    candidate_from_db_root = os.path.abspath(os.path.join(db_root, p_clean))
    if os.path.exists(candidate_from_db_root):
        return candidate_from_db_root

    return os.path.abspath(os.path.join(os.path.dirname(rec_file_path), p_clean))


def _parse_quality_score(row: Dict[str, Any]) -> float:
    """从记录行中解析质量分数，失败时返回负无穷。"""
    q_raw = str(row.get('quality_score', '')).strip()
    if q_raw:
        try:
            return float(q_raw)
        except Exception:
            pass

    q_percent = str(row.get('quality_score_percent', '')).strip().replace('%', '')
    if q_percent:
        try:
            return float(q_percent) / 100.0
        except Exception:
            pass

    return float('-inf')


def _export_best_faces_by_cluster(
    rows: List[Dict[str, Any]],
    cluster_field: str,
    output_dir: str,
    rec_file_path: str,
    db_root: str,
    best_face_dir_name: str,
    index_file_name: str,
    rec_file_field: Optional[str] = None,
) -> Tuple[int, str, str]:
    """按簇导出最高质量人脸，并输出索引CSV。"""
    if not rows:
        return 0, "", ""

    best_by_cluster: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cluster_id = str(row.get(cluster_field, '')).strip()
        if not cluster_id:
            continue

        score = _parse_quality_score(row)
        if np.isneginf(score):
            continue

        prev = best_by_cluster.get(cluster_id)
        if prev is None or score > prev['score']:
            best_by_cluster[cluster_id] = {
                'row': row,
                'score': score,
            }

    if not best_by_cluster:
        return 0, "", ""

    best_face_dir = os.path.join(output_dir, best_face_dir_name)
    os.makedirs(best_face_dir, exist_ok=True)

    index_path = os.path.join(output_dir, index_file_name)
    saved_count = 0

    try:
        with open(index_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                cluster_field,
                'quality_score',
                'source_image_path',
                'saved_image_path',
            ])

            for cluster_id in sorted(best_by_cluster.keys(), key=lambda x: int(x) if x.isdigit() else x):
                item = best_by_cluster[cluster_id]
                row = item['row']
                score = item['score']

                row_rec_file_path = rec_file_path
                if rec_file_field:
                    row_rec_file_path = str(row.get(rec_file_field, rec_file_path))
                src = _resolve_image_path(row.get('image_path', ''), row_rec_file_path, db_root)
                if not src or not os.path.exists(src):
                    continue

                ext = os.path.splitext(src)[1] or '.jpg'
                dst_name = f"cluster_{cluster_id}_best_q{score:.4f}{ext}"
                dst = os.path.join(best_face_dir, dst_name)
                try:
                    shutil.copy2(src, dst)
                    saved_count += 1
                    writer.writerow([
                        cluster_id,
                        f"{score:.6f}",
                        src,
                        dst,
                    ])
                except Exception:
                    continue
    except Exception:
        return 0, "", best_face_dir

    return saved_count, index_path, best_face_dir


def refine_persona_ids_with_clustering(
    rec_path: str,
    output_dir: str,
    metric: str = 'cosine',
    threshold: float = 0.45,
    min_cluster_size: int = 2,
    embedding_mode: str = 'half_body',
) -> Tuple[int, int]:
    """基于已保存的embedding进行离线聚类精修，输出精修后的记录文件。"""
    if not os.path.exists(rec_path):
        return 0, 0

    try:
        with open(rec_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        logger.warning(f"读取记录文件失败，跳过聚类精修: {e}")
        return 0, 0

    if not rows:
        return 0, 0

    pid_to_embeddings: Dict[int, List[np.ndarray]] = {}
    for row in rows:
        try:
            pid = int(str(row.get('persona_id', '')).strip())
        except Exception:
            continue

        emb_path = _resolve_embedding_path(row.get('embedding_path', ''), rec_path, output_dir)
        if not emb_path or not os.path.exists(emb_path):
            continue
        if not _is_embedding_path_for_mode(emb_path, embedding_mode):
            continue

        try:
            emb = np.asarray(np.load(emb_path), dtype=np.float32).reshape(-1)
        except Exception:
            continue

        norm = float(np.linalg.norm(emb))
        if emb.size == 0 or norm <= 0:
            continue

        emb = emb / norm
        pid_to_embeddings.setdefault(pid, []).append(emb)

    pids = sorted(pid_to_embeddings.keys())
    clusters: Dict[int, List[int]] = {}
    pid_to_refined: Dict[int, int] = {}

    if len(pids) < 2:
        for idx, pid in enumerate(pids):
            clusters[idx] = [pid]
            pid_to_refined[pid] = pid
    else:
        proto_vecs: List[np.ndarray] = []
        for pid in pids:
            v = np.mean(np.stack(pid_to_embeddings[pid], axis=0), axis=0)
            v = v / (np.linalg.norm(v) + 1e-12)
            proto_vecs.append(v.astype(np.float32))

        proto_mat = np.stack(proto_vecs, axis=0)
        n = len(pids)
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra = _find(a)
            rb = _find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            vi = proto_mat[i]
            for j in range(i + 1, n):
                vj = proto_mat[j]
                if metric == 'euclidean':
                    score = float(np.linalg.norm(vi - vj))
                    is_match = score <= threshold
                else:
                    score = float(np.dot(vi, vj))
                    is_match = score >= threshold

                if is_match:
                    _union(i, j)

        for idx, pid in enumerate(pids):
            clusters.setdefault(_find(idx), []).append(pid)

        for members in clusters.values():
            members = sorted(members)
            canonical = members[0]
            for pid in members:
                pid_to_refined[pid] = canonical

    refined_persona_count = len(set(pid_to_refined.values()))
    merged_persona_count = len(pids) - refined_persona_count

    mapping_path = os.path.join(output_dir, cfg.CLUSTER_MAPPING_FILE)
    try:
        with open(mapping_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['old_persona_id', 'refined_persona_id'])
            for pid in pids:
                writer.writerow([pid, pid_to_refined.get(pid, pid)])
    except Exception as e:
        logger.warning(f"写入聚类映射失败: {e}")

    clustered_rec_path = os.path.join(output_dir, cfg.CLUSTERED_RECORD_FILE)
    clustered_rows: List[Dict[str, Any]] = []
    try:
        fieldnames = list(rows[0].keys())
        if 'refined_persona_id' not in fieldnames:
            fieldnames.append('refined_persona_id')

        with open(clustered_rec_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row_copy = dict(row)
                try:
                    pid = int(str(row.get('persona_id', '')).strip())
                except Exception:
                    pid = None
                row_copy['refined_persona_id'] = pid_to_refined.get(pid, pid if pid is not None else '')
                writer.writerow(row_copy)
                clustered_rows.append(row_copy)
    except Exception as e:
        logger.warning(f"写入聚类后记录文件失败: {e}")

    best_face_count, best_face_index_path, best_face_dir = _export_best_faces_by_cluster(
        rows=clustered_rows,
        cluster_field='refined_persona_id',
        output_dir=output_dir,
        rec_file_path=rec_path,
        db_root=output_dir,
        best_face_dir_name=cfg.CLUSTER_BEST_FACE_DIR,
        index_file_name=cfg.CLUSTER_BEST_FACE_INDEX_FILE,
    )

    valid_clusters = 0
    for members in clusters.values():
        if len(members) >= max(2, int(min_cluster_size)):
            valid_clusters += 1

    logger.info(
        f"🧩 聚类精修完成: 原persona={len(pids)} -> 精修后={refined_persona_count}, "
        f"合并={merged_persona_count}, 阈值={threshold:.3f}({metric}), 有效簇={valid_clusters}"
    )
    logger.info(f"   - 聚类映射: {mapping_path}")
    logger.info(f"   - 精修记录: {clustered_rec_path}")
    if best_face_count > 0:
        logger.info(f"   - 每簇最佳人脸: {best_face_dir} ({best_face_count} 张)")
        logger.info(f"   - 最佳人脸索引: {best_face_index_path}")

    return merged_persona_count, refined_persona_count


def _build_cluster_mapping(
    keys: List[str],
    proto_mat: np.ndarray,
    metric: str,
    threshold: float,
) -> Tuple[Dict[str, str], int]:
    """对原型向量做阈值连通分量聚类，返回 key 到 canonical_key 的映射。"""
    n = len(keys)
    if n == 0:
        return {}, 0

    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra = _find(a)
        rb = _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        vi = proto_mat[i]
        for j in range(i + 1, n):
            vj = proto_mat[j]
            if metric == 'euclidean':
                score = float(np.linalg.norm(vi - vj))
                is_match = score <= threshold
            else:
                score = float(np.dot(vi, vj))
                is_match = score >= threshold

            if is_match:
                _union(i, j)

    clusters: Dict[int, List[int]] = {}
    for idx in range(n):
        clusters.setdefault(_find(idx), []).append(idx)

    key_to_canonical: Dict[str, str] = {}
    for members in clusters.values():
        member_keys = sorted(keys[i] for i in members)
        canonical_key = member_keys[0]
        for k in member_keys:
            key_to_canonical[k] = canonical_key

    merged_count = n - len(set(key_to_canonical.values()))
    return key_to_canonical, merged_count


def refine_global_persona_ids_with_clustering(
    output_root: str,
    metric: str = 'cosine',
    threshold: float = 0.45,
    min_cluster_size: int = 2,
    embedding_mode: str = 'half_body',
) -> Tuple[int, int, int]:
    """
    跨视频全局聚类精修。

    在 output_root 下扫描全部 face_records_frontal.txt，
    将每个“视频内persona”视为一个节点，使用原型向量聚类并输出全局ID。

    Returns:
        (merged_count, global_persona_count, processed_record_file_count)
    """
    if not output_root or not os.path.isdir(output_root):
        return 0, 0, 0

    rec_files: List[str] = []
    for root, _, files in os.walk(output_root):
        if 'face_records_frontal.txt' in files:
            rec_files.append(os.path.join(root, 'face_records_frontal.txt'))

    if not rec_files:
        return 0, 0, 0

    source_to_embeddings: Dict[str, List[np.ndarray]] = {}
    source_meta: Dict[str, Dict[str, Any]] = {}
    file_rows: Dict[str, List[Dict[str, str]]] = {}

    for rec_file in rec_files:
        try:
            with open(rec_file, 'r', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            logger.warning(f"读取记录文件失败(全局聚类跳过该文件): {rec_file}, 错误: {e}")
            continue

        if not rows:
            continue

        file_rows[rec_file] = rows
        video_dir = os.path.relpath(os.path.dirname(rec_file), output_root)

        for row in rows:
            try:
                pid = int(str(row.get('persona_id', '')).strip())
            except Exception:
                continue

            emb_path = _resolve_embedding_path(row.get('embedding_path', ''), rec_file, output_root)
            if not emb_path or not os.path.exists(emb_path):
                continue
            if not _is_embedding_path_for_mode(emb_path, embedding_mode):
                continue

            try:
                emb = np.asarray(np.load(emb_path), dtype=np.float32).reshape(-1)
            except Exception:
                continue

            norm = float(np.linalg.norm(emb))
            if emb.size == 0 or norm <= 0:
                continue

            emb = emb / norm
            source_key = f"{video_dir}::{pid}"
            source_to_embeddings.setdefault(source_key, []).append(emb)
            source_meta[source_key] = {'video_dir': video_dir, 'persona_id': pid}

    if not source_to_embeddings:
        return 0, 0, len(file_rows)

    keys = sorted(source_to_embeddings.keys())
    if len(keys) < 2:
        key_to_canonical = {k: k for k in keys}
        merged_count = 0
    else:
        proto_vecs: List[np.ndarray] = []
        for k in keys:
            v = np.mean(np.stack(source_to_embeddings[k], axis=0), axis=0)
            v = v / (np.linalg.norm(v) + 1e-12)
            proto_vecs.append(v.astype(np.float32))
        proto_mat = np.stack(proto_vecs, axis=0)
        key_to_canonical, merged_count = _build_cluster_mapping(keys, proto_mat, metric, threshold)

    canonical_keys = sorted(set(key_to_canonical.values()))
    canonical_to_global_id = {k: i + 1 for i, k in enumerate(canonical_keys)}
    key_to_global_id = {k: canonical_to_global_id[key_to_canonical[k]] for k in keys}

    mapping_path = os.path.join(output_root, cfg.GLOBAL_CLUSTER_MAPPING_FILE)
    try:
        with open(mapping_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'video_dir',
                'old_persona_id',
                'global_refined_persona_id',
                'canonical_video_dir',
                'canonical_persona_id',
            ])
            for k in keys:
                meta = source_meta[k]
                c_key = key_to_canonical[k]
                c_meta = source_meta[c_key]
                writer.writerow([
                    meta['video_dir'],
                    meta['persona_id'],
                    key_to_global_id[k],
                    c_meta['video_dir'],
                    c_meta['persona_id'],
                ])
    except Exception as e:
        logger.warning(f"写入全局聚类映射失败: {e}")

    global_rows: List[Dict[str, Any]] = []
    best_face_rows: List[Dict[str, Any]] = []
    for rec_file, rows in file_rows.items():
        video_dir = os.path.relpath(os.path.dirname(rec_file), output_root)
        out_file = os.path.join(os.path.dirname(rec_file), cfg.PER_VIDEO_GLOBAL_CLUSTERED_RECORD_FILE)

        try:
            fieldnames = list(rows[0].keys())
            if 'global_refined_persona_id' not in fieldnames:
                fieldnames.append('global_refined_persona_id')

            with open(out_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    row_copy = dict(row)
                    try:
                        pid = int(str(row.get('persona_id', '')).strip())
                        source_key = f"{video_dir}::{pid}"
                        gid = key_to_global_id.get(source_key, '')
                    except Exception:
                        gid = ''
                    row_copy['global_refined_persona_id'] = gid
                    writer.writerow(row_copy)
                    row_copy['video_dir'] = video_dir
                    global_rows.append(row_copy)

                    row_for_best = dict(row_copy)
                    row_for_best['__rec_file_path'] = rec_file
                    best_face_rows.append(row_for_best)
        except Exception as e:
            logger.warning(f"写入视频级全局聚类记录失败: {out_file}, 错误: {e}")

    global_file = os.path.join(output_root, cfg.GLOBAL_CLUSTERED_RECORD_FILE)
    if global_rows:
        try:
            fieldnames = list(global_rows[0].keys())
            with open(global_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(global_rows)
        except Exception as e:
            logger.warning(f"写入全局聚类总表失败: {e}")

    best_face_count, best_face_index_path, best_face_dir = _export_best_faces_by_cluster(
        rows=best_face_rows,
        cluster_field='global_refined_persona_id',
        output_dir=output_root,
        rec_file_path=output_root,
        db_root=output_root,
        best_face_dir_name=cfg.GLOBAL_CLUSTER_BEST_FACE_DIR,
        index_file_name=cfg.GLOBAL_CLUSTER_BEST_FACE_INDEX_FILE,
        rec_file_field='__rec_file_path',
    )

    # 仅用于统计展示
    canonical_count = len(canonical_keys)
    valid_clusters = 0
    cluster_sizes: Dict[str, int] = {}
    for k in keys:
        c_key = key_to_canonical[k]
        cluster_sizes[c_key] = cluster_sizes.get(c_key, 0) + 1
    for _, sz in cluster_sizes.items():
        if sz >= max(2, int(min_cluster_size)):
            valid_clusters += 1

    logger.info(
        f"🌐 全局聚类精修完成: 原persona={len(keys)} -> 全局ID={canonical_count}, "
        f"合并={merged_count}, 阈值={threshold:.3f}({metric}), 有效簇={valid_clusters}"
    )
    logger.info(f"   - 全局映射: {mapping_path}")
    logger.info(f"   - 全局总表: {global_file}")
    if best_face_count > 0:
        logger.info(f"   - 每簇最佳人脸: {best_face_dir} ({best_face_count} 张)")
        logger.info(f"   - 最佳人脸索引: {best_face_index_path}")

    return merged_count, canonical_count, len(file_rows)


def load_existing_embeddings_to_deduper(deduper: Deduper, db_root: str, embedding_mode: str = 'half_body') -> int:
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
                    if not _is_embedding_path_for_mode(emb_path, embedding_mode):
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
                if not _is_embedding_path_for_mode(emb_path, embedding_mode):
                    continue
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
        
        # 简化版人脸验证器（用于实时质量评分）
        # 配置项：质量阈值、置信度阈值、严格模式
        self.face_validator = SimpleFaceValidator(
            quality_threshold=getattr(args, 'quality_threshold', 0.2),
            confidence_threshold=getattr(args, 'confidence_threshold', 0.4),
            strict_mode=getattr(args, 'strict_mode', False)
        )
        
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
        
        # 构建两个查找表：kps 和 embedding（避免混用不同来源的特征）
        bbox_to_kps = {}
        bbox_to_embedding = {}
        if frame_detections:
            for det in frame_detections:
                if det:
                    bbox = det.get('bbox')
                    if bbox:
                        # 以bbox作为key（四舍五入到整数）
                        bbox_key = tuple(map(int, bbox))
                        if det.get('kps') is not None:
                            bbox_to_kps[bbox_key] = det.get('kps')
                        if det.get('embedding') is not None:
                            bbox_to_embedding[bbox_key] = det.get('embedding')
        
        # ============= 帧级去重：收集本帧的所有人脸embedding =============
        # 同一帧中可能出现多个人脸，如果相似度高应该去重
        frame_embeddings = []  # [(tid, bbox, kps, emb, quality_result), ...]
        
        # ============= 帧级过滤：预先计算每个人脸的尺寸，找出最大人脸 =============
        # 目的：在同一帧中，只保存尺寸最大的人脸，跳过其他较小的人脸（减少冗余输出）
        face_sizes = {}  # {j: (size, bbox)} - 记录每个人脸的尺寸
        max_size_idx = -1
        max_size = 0
        
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[j])
            width = x2 - x1
            height = y2 - y1
            size = width * height  # 用面积作为尺寸的度量
            face_sizes[j] = (size, (x1, y1, x2, y2))
            
            # 找出最大的人脸
            if size > max_size:
                max_size = size
                max_size_idx = j
        
        logger.debug(f"📐 本帧找到 {len(xyxy)} 个人脸，最大人脸索引={max_size_idx}，尺寸={max_size}px²")
        
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
            embedding_mode = str(getattr(self.args, 'embedding_mode', 'half_body')).strip().lower()
            need_half_body = bool(
                embedding_mode == 'half_body' or
                getattr(self.args, 'save_half_body', True) or
                getattr(self.args, 'use_context_reid', False)
            )
            half_body_crop = None
            context_hist = None
            if need_half_body:
                half_body_crop = extract_half_body_crop(
                    frame,
                    (x1, y1, x2, y2),
                    top_expand=float(getattr(self.args, 'half_body_top_expand', cfg.HALF_BODY_TOP_EXPAND)),
                    bottom_expand=float(getattr(self.args, 'half_body_bottom_expand', cfg.HALF_BODY_BOTTOM_EXPAND)),
                    side_expand=float(getattr(self.args, 'half_body_side_expand', cfg.HALF_BODY_SIDE_EXPAND)),
                )
            if getattr(self.args, 'use_context_reid', False):
                context_hist = extract_apparel_context_hist(
                    half_body_crop,
                    hist_bins=int(getattr(self.args, 'context_hist_bins', cfg.CONTEXT_HIST_BINS)),
                )
            match_metadata = build_context_match_metadata(self.args, context_hist)
            
            # ============= 步骤 3 & 4: 提取关键点和评估质量 =============
            # 策略：优先使用frame-level detect的kps和embedding（避免混用不同来源特征）
            kps = None
            det_embedding = None
            
            # 尝试从frame-level detection的结果中找到匹配的kps和embedding
            bbox_key = (x1, y1, x2, y2)
            if bbox_key in bbox_to_kps:
                kps = bbox_to_kps[bbox_key]
            else:
                # 如果没有精确匹配，尝试近似匹配（允许一些像素偏差）
                for stored_bbox, stored_kps in bbox_to_kps.items():
                    if (abs(stored_bbox[0]-x1) < 5 and abs(stored_bbox[1]-y1) < 5 and
                        abs(stored_bbox[2]-x2) < 5 and abs(stored_bbox[3]-y2) < 5):
                        kps = stored_kps
                        # 同时尝试获取对应的embedding
                        if stored_bbox in bbox_to_embedding:
                            det_embedding = bbox_to_embedding[stored_bbox]
                        break
            
            # 如果在kps查询中没有找到embedding，再尝试直接的bbox_key查询
            if det_embedding is None and bbox_key in bbox_to_embedding:
                det_embedding = bbox_to_embedding[bbox_key]
            
            # 重要：不再进行第二次检测以获取embedding
            #（第二次检测会导致不同尺寸的特征，与第一次frame-level detection的特征不可比较）
            
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
            
            # ============= 帧级过滤：只处理同一帧中的最大人脸 =============
            # 目的：在一帧中出现多个人脸时，只处理尺寸最大的那个，其他跳过
            if j != max_size_idx and len(xyxy) > 1:
                size, bbox = face_sizes[j]
                logger.debug(f"⏭️  跳过较小人脸: track{tid} 尺寸={size}px² (本帧最大={max_size}px², 最大索引={max_size_idx})")
                continue  # 跳过这个不是最大的人脸
            
            # 对齐人脸（使用之前获取的kps，或从检测中获取）
            aligned = align_face(face_crop, kps)
            
            # ============= 步骤 7: 提取 Embedding =============
            # 默认使用半身区域embedding（衣物上下文）；face模式时回退到人脸embedding。
            # 通过 cluster_feature_mode 决定特征提取方式（arcface/reid/hsv）
            emb = None
            if embedding_mode == 'half_body':
                emb = extract_half_body_embedding(
                    half_body_crop,
                    embedding_mode=embedding_mode,
                    use_reid=getattr(self.args, 'use_reid_feature', True),
                    hist_bins=int(getattr(self.args, 'context_hist_bins', cfg.CONTEXT_HIST_BINS)),
                    cluster_feature_mode=getattr(self.args, 'cluster_feature', 'reid'),
                )
                if emb is None:
                    logger.debug(f"⚠️  跳过无半身embedding的人脸 (tid={tid})")
                    continue
                logger.debug(f"📊 使用half-body embedding (tid={tid}, shape={emb.shape}, feature={getattr(self.args, 'cluster_feature', 'reid')})")
            else:
                if det_embedding is not None:
                    emb = np.asarray(det_embedding, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    logger.debug(f"📊 使用face embedding (tid={tid}, shape={emb.shape})")
                else:
                    logger.debug(f"⚠️  跳过无face embedding的人脸 (tid={tid})")
                    continue

            # 使用半身embedding时，不再叠加context二次融合，避免重复计分
            match_metadata_for_find = {} if embedding_mode == 'half_body' else match_metadata
            
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
                # ================ 验证步骤：画质评分 + 置信度检查 ================
                # 在进行去重之前，确保人脸质量足够好
                is_valid, computed_quality, reason = self.face_validator.validate_face(
                    face_crop=face_crop,
                    confidence=conf,
                    embedding=emb
                )
                
                if not is_valid:
                    logger.debug(f"⚠️  人脸验证失败: track={tid} | frame={frame_count} | {reason}")
                    continue  # 跳过这个人脸，不进行后续处理
                
                logger.debug(f"✅ 人脸验证通过: track={tid} | 画质={computed_quality:.4f}")
                
                # 更新质量分数为计算出的分数
                actual_quality_score = max(quality_result.quality_score, computed_quality)
                
                rec.frontal_count += 1
                
                # 更新最佳正脸
                if actual_quality_score > rec.best_quality_score:
                    rec.best_quality_score = actual_quality_score
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
                    'quality_score': actual_quality_score,
                    'aligned': aligned.copy() if aligned is not None else face_crop.copy(),
                    'rec': rec
                })
            
            # ================ 无论如何都使用 TrackHistory 检查连续重复 ================
            # 对所有正脸调用 track_history.update，记录跟踪历史信息
            if quality_result.is_frontal and emb is not None:
                should_save_this_frame = self.track_history.update(tid, frame_count, actual_quality_score)
                logger.debug(f"📹 正在处理轨迹ID={tid} | 帧={frame_count} | 质量分数={actual_quality_score:.4f} | "
                           f"should_save={should_save_this_frame}")
            
            # 保存策略: first - 第一个正脸立即保存（全局唯一策略）
            if not rec.first_saved:
                if quality_result.is_frontal and emb is not None:
                    # ================ 连续重复检查：避免保存同一person的重复帧 ================
                    # 根据 track_history 的结果决定是否保存
                    if not should_save_this_frame:
                        logger.info(f"⏭️  跳过连续重复帧: track{tid} 在frame{frame_count} 连续出现，质量不如最好帧")
                        continue  # 跳过这一帧的保存
                    
                    # ================ 改进方案1：在first策略中启用多帧聚合 ================
                    # 使用多帧汇总的embedding进行去重，提高相似度准确性
                    try:
                        if len(rec.embeddings) > 1:
                            agg_emb = np.mean(np.stack(rec.embeddings, axis=0), axis=0)
                            # 检查聚合embedding的合法性
                            if agg_emb is None or agg_emb.size == 0:
                                logger.warning(f"⚠️  聚合embedding无效，降级到单帧: track{tid}")
                                pid = self.deduper.find_match(emb, metadata=match_metadata_for_find, debug=True)
                            else:
                                logger.debug(f"📊 First策略：使用聚合embedding ({len(rec.embeddings)}帧的平均，norm={np.linalg.norm(agg_emb):.6f})")
                                pid = self.deduper.find_match(agg_emb, metadata=match_metadata_for_find, debug=True)
                        else:
                            logger.debug(f"📊 First策略：仅有单帧，使用单帧embedding（norm={np.linalg.norm(emb):.6f}）")
                            pid = self.deduper.find_match(emb, metadata=match_metadata_for_find, debug=True)
                    except Exception as e:
                        logger.error(f"❌ embedding聚合过程出错: track{tid}, error={e}")
                        # 降级处理：使用单帧embedding
                        try:
                            pid = self.deduper.find_match(emb, metadata=match_metadata_for_find, debug=True)
                        except Exception as e2:
                            logger.error(f"❌ 单帧embedding匹配失败: track{tid}, error={e2}")
                            pid = None
                    if pid is None:
                        add_metadata = {
                            'yaw': quality_result.yaw,
                            'pitch': quality_result.pitch,
                            'roll': quality_result.roll,
                            'embedding_mode': embedding_mode,
                        }
                        if embedding_mode == 'face' and context_hist is not None:
                            add_metadata['context_hist'] = context_hist
                        pid = self.deduper.add(emb, add_metadata)
                        
                        # ================ 健壮性增强：保存前验证所有数据 ================
                        # 检查所有必要数据的合法性
                        if pid < 0:
                            logger.error(f"❌ 添加新ID失败: pid={pid} | track={tid}")
                            continue
                        
                        if face_crop is None or face_crop.size == 0:
                            logger.error(f"❌ 人脸裁剪无效: track={tid}")
                            continue
                        
                        if frame is None or frame.size == 0:
                            logger.error(f"❌ 原始帧无效: track={tid}")
                            continue
                        
                        if emb is None or emb.size == 0:
                            logger.error(f"❌ embedding数据无效: pid={pid:05d} | track={tid}")
                            continue
                        
                        # 半身优先保存，人脸仅作为参考图
                        os.makedirs(os.path.join(output_dir, 'embeddings'), exist_ok=True)
                        os.makedirs(os.path.join(output_dir, 'debug_frames'), exist_ok=True)  # 调试用
                        os.makedirs(os.path.join(output_dir, 'half_body'), exist_ok=True)
                        os.makedirs(os.path.join(output_dir, 'face_ref'), exist_ok=True)
                        half_body_path = os.path.join(output_dir, 'half_body', f"face_{pid:05d}_track{tid}_f{frame_count}.jpg")
                        face_ref_path = os.path.join(output_dir, 'face_ref', f"face_{pid:05d}_track{tid}_f{frame_count}.jpg")
                        if embedding_mode == 'half_body':
                            img_path = half_body_path
                            emb_path = os.path.join(output_dir, 'embeddings', f"face_{pid:05d}_track{tid}_f{frame_count}_body.npy")
                        else:
                            img_path = face_ref_path
                            emb_path = os.path.join(output_dir, 'embeddings', f"face_{pid:05d}_track{tid}_f{frame_count}.npy")
                        frame_path = os.path.join(output_dir, 'debug_frames', f"frame_{pid:05d}_track{tid}_f{frame_count}.jpg")  # 保存原始帧
                        
                        try:
                            half_body_saved = True
                            if half_body_crop is not None and half_body_crop.size > 0:
                                half_body_saved = cv2.imwrite(half_body_path, half_body_crop)

                            face_saved = cv2.imwrite(face_ref_path, face_crop)
                            # 保存原始帧截图（用于对比分析）
                            frame_saved = cv2.imwrite(frame_path, frame)
                            
                            if embedding_mode == 'half_body' and not half_body_saved:
                                logger.error(f"❌ 保存半身主图失败: {half_body_path}")
                                continue

                            if not face_saved:
                                logger.error(f"❌ 保存人脸参考图失败: {face_ref_path}, size={face_crop.shape if face_crop is not None else 'None'}")
                                try:
                                    if os.path.exists(half_body_path) and embedding_mode == 'half_body':
                                        os.remove(half_body_path)
                                except:
                                    pass
                                continue
                            
                            if not frame_saved:
                                logger.error(f"❌ 保存原始帧失败: {frame_path}, size={frame.shape if frame is not None else 'None'}")
                                try:
                                    if os.path.exists(img_path):
                                        os.remove(img_path)
                                    if os.path.exists(face_ref_path):
                                        os.remove(face_ref_path)
                                    if os.path.exists(half_body_path):
                                        os.remove(half_body_path)
                                except:
                                    pass
                                continue

                            if not half_body_saved:
                                logger.warning(f"⚠️  半身图保存失败: {half_body_path}")
                            
                            # 保存 embedding（应该来自 InsightFace，质量更高）
                            if emb is not None and emb.size > 0:
                                try:
                                    # ================ 健壮性增强：embedding保存验证 ================
                                    np.save(emb_path, emb.astype(np.float32))
                                    
                                    # 验证embedding是否成功保存
                                    saved_emb = np.load(emb_path)
                                    if saved_emb is None or saved_emb.size != emb.size:
                                        logger.error(f"❌ Embedding保存或读取失败: pid={pid:05d} | track={tid}")
                                        try:
                                            os.remove(emb_path)
                                            os.remove(img_path)
                                            os.remove(frame_path)
                                            if os.path.exists(face_ref_path):
                                                os.remove(face_ref_path)
                                            if os.path.exists(half_body_path):
                                                os.remove(half_body_path)
                                        except:
                                            pass
                                        continue
                                    
                                    saved_count += 1
                                    rec.first_saved = True
                                    
                                    # 保存质量检测详细信息到文本文件
                                    try:
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
                                    except Exception as e:
                                        logger.warning(f"⚠️  保存调试信息失败: {e}")
                                    
                                    logger.info(f"✅ 保存首帧正脸: pid={pid:05d} | track={tid} | frame={frame_count} | 质量={quality_result.quality_score:.4f} | "
                                              f"姿态(Y={quality_result.yaw:.1f}°, P={quality_result.pitch:.1f}°, R={quality_result.roll:.1f}°) | emb_dim={emb.size}")
                                    
                                    try:
                                        with open(rec_path, 'a', encoding='utf-8') as f:
                                            f.write(f"{pid},{tid},{time_str},{timestamp:.2f},{quality_result.quality_score:.4f},{quality_result.quality_score*100:.2f}%,{img_path},{emb_path},{frame_count}\n")
                                    except Exception as e:
                                        logger.error(f"❌ 写入记录文件失败: {e}")
                                        # 即使记录失败，也要继续处理
                                
                                except Exception as e:
                                    logger.error(f"❌ Embedding保存过程出错: pid={pid:05d} | track={tid} | error={e}")
                                    try:
                                        os.remove(emb_path)
                                        os.remove(img_path)
                                        os.remove(frame_path)
                                        if os.path.exists(face_ref_path):
                                            os.remove(face_ref_path)
                                        if os.path.exists(half_body_path):
                                            os.remove(half_body_path)
                                    except:
                                        pass
                                    continue
                            else:
                                logger.warning(f"⚠️  embedding 无效或为空: pid={pid:05d} | track={tid} | size={emb.size if emb is not None else 'None'}")
                                try:
                                    os.remove(img_path)
                                    os.remove(frame_path)
                                    if os.path.exists(face_ref_path):
                                        os.remove(face_ref_path)
                                    if os.path.exists(half_body_path):
                                        os.remove(half_body_path)
                                except:
                                    pass
                        except Exception as e:
                            logger.error(f"❌ 保存人脸过程出错: track={tid}, error={e}")
                            try:
                                os.remove(img_path)
                                os.remove(frame_path)
                                if os.path.exists(face_ref_path):
                                    os.remove(face_ref_path)
                                if os.path.exists(half_body_path):
                                    os.remove(half_body_path)
                            except:
                                pass
        
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
    


def process_video(video_path: str, output_dir: str, args):
    """处理单个视频"""
    os.makedirs(output_dir, exist_ok=True)

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
    
    # 🆕 根据视频分辨率动态计算所有尺寸阈值
    size_thresholds = cfg.calculate_size_thresholds(H, W, cfg.MIN_FACE_SIZE_RATIO)
    # 更新全局配置中的相关参数，供后续处理使用
    cfg.BYTETRACK_CONFIG['min_box_area'] = size_thresholds['min_box_area']
    cfg.FACE_SAVE_VALIDATOR['min_face_width'] = size_thresholds['min_face_width']
    cfg.FACE_SAVE_VALIDATOR['min_face_height'] = size_thresholds['min_face_height']
    # 动态设置为全局属性，供 face_dedup_utils 使用
    cfg.MIN_FACE_SIZE = size_thresholds['min_face_size']
    cfg.MIN_SAVE_FACE_SIZE = size_thresholds['min_save_face_size']
    logger.info(f"📊 视频分辨率: {W}×{H}, MIN_FACE_SIZE_RATIO={cfg.MIN_FACE_SIZE_RATIO}")
    logger.info(f"   动态阈值: min_face_size={size_thresholds['min_face_size']}px, "
                f"min_box_area={size_thresholds['min_box_area']}px², "
                f"min_face_width/height={size_thresholds['min_face_width']}px")
    
    # 🔄 在计算动态阈值后创建提取器
    extractor = FrontalFaceExtractor(args)

    # 加载历史 embedding 库，支持和已有人脸ID进行匹配
    if getattr(args, 'reuse_embedding_db', True):
        loaded = load_existing_embeddings_to_deduper(
            extractor.deduper,
            args.embedding_db_dir,
            embedding_mode=args.embedding_mode,
        )
        if loaded > 0:
            logger.info(f"已加载历史embedding库: {loaded} 个persona (mode={args.embedding_mode}, db: {args.embedding_db_dir})")
    
    # 记录文件
    rec_path = os.path.join(output_dir, 'face_records_frontal.txt')
    if not os.path.exists(rec_path) or os.path.getsize(rec_path) == 0:
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write('persona_id,track_id,time,timestamp,quality_score,quality_score_percent,image_path,embedding_path,frame_count\n')
    
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
                
                previous_active = current_active
                
                # 定期清理过期的 track_history 记录（避免内存堆积）
                if frame_count % 50 == 0:
                    cleaned = extractor.track_history.cleanup(frame_count)
                    if cleaned > 0:
                        logger.debug(f"🧹 TrackHistory cleanup: 清理了{cleaned}个过期track记录")
            else:
                # 不使用跟踪模式：逐帧检测
                dets = extractor.det.detect(frame, conf_threshold=args.conf)
                for d in dets:
                    # 先裁剪人脸区域，以便进行物理质量检查
                    x1, y1, x2, y2 = d['bbox']
                    face_crop = frame[y1:y2, x1:x2]
                    embedding_mode = str(getattr(args, 'embedding_mode', 'half_body')).strip().lower()
                    need_half_body = bool(
                        embedding_mode == 'half_body' or
                        getattr(args, 'save_half_body', True) or
                        getattr(args, 'use_context_reid', False)
                    )
                    half_body_crop = None
                    context_hist = None
                    if need_half_body:
                        half_body_crop = extract_half_body_crop(
                            frame,
                            (x1, y1, x2, y2),
                            top_expand=float(getattr(args, 'half_body_top_expand', cfg.HALF_BODY_TOP_EXPAND)),
                            bottom_expand=float(getattr(args, 'half_body_bottom_expand', cfg.HALF_BODY_BOTTOM_EXPAND)),
                            side_expand=float(getattr(args, 'half_body_side_expand', cfg.HALF_BODY_SIDE_EXPAND)),
                        )
                    if getattr(args, 'use_context_reid', False):
                        context_hist = extract_apparel_context_hist(
                            half_body_crop,
                            hist_bins=int(getattr(args, 'context_hist_bins', cfg.CONTEXT_HIST_BINS)),
                        )
                    match_metadata = build_context_match_metadata(args, context_hist)
                    
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
                    
                    if embedding_mode == 'half_body':
                        emb = extract_half_body_embedding(
                            half_body_crop,
                            embedding_mode=embedding_mode,
                            use_reid=getattr(args, 'use_reid_feature', True),
                            hist_bins=int(getattr(args, 'context_hist_bins', cfg.CONTEXT_HIST_BINS)),
                            cluster_feature_mode=getattr(args, 'cluster_feature', 'reid'),
                        )
                        if emb is None:
                            logger.debug(f"⚠️  跳过无半身embedding的人脸（帧{frame_count}）")
                            continue
                        match_metadata_for_find = {}
                    else:
                        emb = d.get('embedding')
                        if emb is not None:
                            emb = np.asarray(emb, dtype=np.float32)
                            norm = np.linalg.norm(emb)
                            if norm > 0:
                                emb = emb / norm
                        else:
                            logger.debug(f"⚠️  跳过无face embedding的人脸（帧{frame_count}）")
                            continue
                        match_metadata_for_find = match_metadata
                    
                    logger.debug(f"📹 正在处理人脸检测 | 帧={frame_count} | 质量分数={quality_result.quality_score:.4f}")
                    pid = extractor.deduper.find_match(emb, metadata=match_metadata_for_find, debug=False)
                    if pid is None:
                        add_metadata = {
                            'yaw': quality_result.yaw,
                            'pitch': quality_result.pitch,
                            'roll': quality_result.roll,
                            'embedding_mode': embedding_mode,
                        }
                        if embedding_mode == 'face' and context_hist is not None:
                            add_metadata['context_hist'] = context_hist
                        pid = extractor.deduper.add(emb, add_metadata)
                        half_body_path = os.path.join(output_dir, 'half_body', f"face_{pid:05d}_f{frame_count}.jpg")
                        face_ref_path = os.path.join(output_dir, 'face_ref', f"face_{pid:05d}_f{frame_count}.jpg")
                        if embedding_mode == 'half_body':
                            img_path = half_body_path
                            emb_path = os.path.join(output_dir, 'embeddings', f"face_{pid:05d}_f{frame_count}_body.npy")
                        else:
                            img_path = face_ref_path
                            emb_path = os.path.join(output_dir, 'embeddings', f"face_{pid:05d}_f{frame_count}.npy")
                        os.makedirs(os.path.join(output_dir, 'face_ref'), exist_ok=True)
                        os.makedirs(os.path.join(output_dir, 'half_body'), exist_ok=True)
                        
                        half_body_saved = True
                        if half_body_crop is not None and half_body_crop.size > 0:
                            half_body_saved = save_image(half_body_crop, half_body_path)
                        face_saved = save_image(face_crop if face_crop is not None else aligned, face_ref_path)

                        if embedding_mode == 'half_body' and not half_body_saved:
                            continue

                        if face_saved and save_embedding(emb, emb_path):
                            saved_count += 1
                            with open(rec_path, 'a', encoding='utf-8') as f:
                                f.write(f"{pid},-1,{time_str},{timestamp:.2f},{quality_result.quality_score:.4f},{quality_result.quality_score*100:.2f}%,{img_path},{emb_path},{frame_count}\n")
            
            vw.write(frame)
            frame_count += 1
    
    finally:
        pbar.close()
        cap.release()
        vw.release()
        
        # First策略: 所有face都已在处理过程中实时保存
        # 无需额外处理剩余的track

    if getattr(args, 'cluster_refine', False):
        merged_count, refined_count = refine_persona_ids_with_clustering(
            rec_path=rec_path,
            output_dir=output_dir,
            metric=args.metric,
            threshold=args.cluster_threshold,
            min_cluster_size=args.cluster_min_size,
            embedding_mode=args.embedding_mode,
        )
        if merged_count > 0:
            logger.info(f"✅ 已启用聚类精修: 合并 {merged_count} 个重复persona，保留 {refined_count} 个")
        else:
            logger.info("ℹ️  聚类精修未发现可合并的persona")
    
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

    # 姿态阈值
    parser.add_argument('--yaw-threshold', type=float, default=cfg.YAW_THRESHOLD, help='Yaw角度阈值（度）')
    parser.add_argument('--pitch-threshold', type=float, default=cfg.PITCH_THRESHOLD, help='Pitch角度阈值（度）')
    parser.add_argument('--roll-threshold', type=float, default=cfg.ROLL_THRESHOLD, help='Roll角度阈值（度）')

    # 去重参数
    parser.add_argument('--threshold', type=float, default=cfg.DEDUP_THRESHOLD, help='去重相似度阈值')
    parser.add_argument('--metric', type=str, default=cfg.SIMILARITY_METRIC, choices=['cosine', 'euclidean'])
    parser.add_argument('--embedding-mode', type=str, default=cfg.EMBEDDING_MODE,
                       choices=['half_body', 'face'],
                       help='主embedding来源：half_body(默认)/face')

    # ReID 深度特征
    parser.add_argument('--use-reid-feature', dest='use_reid_feature', action='store_true',
                       help='使用 OSNet ReID 深度特征提取半身 embedding（默认开启）')
    parser.add_argument('--no-use-reid-feature', dest='use_reid_feature', action='store_false',
                       help='禁用 ReID 深度特征，回退到 HSV 直方图')
    parser.set_defaults(use_reid_feature=cfg.USE_REID_FEATURE)
    parser.add_argument('--reid-model', type=str, default=None,
                       help='ReID 模型路径（默认使用 config.py 中的 REID_MODEL_PATH）')

    # 聚类特征模式（决定聚类/去重使用哪种特征提取模型）
    parser.add_argument('--cluster-feature', type=str,
                       choices=['arcface', 'reid', 'hsv'],
                       default=cfg.CLUSTER_FEATURE_MODE,
                       help='聚类特征模式: arcface(ArcFace人脸,推荐)/reid(OSNet半身)/hsv(HSV直方图). '
                            'arcface 适合精确判断同一人，reid 适合区分穿搭不同的人（默认: reid）')

    # 聚类精修参数（第二阶段）
    parser.add_argument('--cluster-refine', dest='cluster_refine', action='store_true',
                       help='处理完成后执行离线聚类精修（默认开启）')
    parser.add_argument('--no-cluster-refine', dest='cluster_refine', action='store_false',
                       help='关闭离线聚类精修')
    parser.set_defaults(cluster_refine=cfg.ENABLE_CLUSTER_REFINEMENT)
    parser.add_argument('--cluster-threshold', type=float, default=None,
                       help='聚类阈值（未设置时按metric自动选择默认值）')
    parser.add_argument('--cluster-min-size', type=int, default=cfg.CLUSTER_MIN_SIZE,
                       help='统计有效簇的最小规模（仅用于日志统计）')

    # 全局聚类精修参数（跨视频）
    parser.add_argument('--global-cluster-refine', dest='global_cluster_refine', action='store_true',
                       help='全部视频处理后执行跨视频全局聚类精修（默认开启）')
    parser.add_argument('--no-global-cluster-refine', dest='global_cluster_refine', action='store_false',
                       help='关闭跨视频全局聚类精修')
    parser.set_defaults(global_cluster_refine=cfg.ENABLE_GLOBAL_CLUSTER_REFINEMENT)
    parser.add_argument('--global-cluster-threshold', type=float, default=None,
                       help='全局聚类阈值（未设置时按metric自动选择默认值）')
    parser.add_argument('--global-cluster-min-size', type=int, default=cfg.GLOBAL_CLUSTER_MIN_SIZE,
                       help='全局聚类统计有效簇的最小规模（仅用于日志统计）')

    # 人脸验证参数（新增）
    parser.add_argument('--quality-threshold', type=float, default=0.2,
                       help='画质评分最低阈值 (0.0-0.7，默认0.2)')
    parser.add_argument('--confidence-threshold', type=float, default=0.4,
                       help='检测置信度最低阈值 (0.3-0.7，默认0.4)')
    parser.add_argument('--strict-mode', dest='strict_mode', action='store_true',
                       help='严格模式：使用更高的验证阈值')

    # 半身上下文匹配（衣物辅助）
    parser.add_argument('--use-context-reid', dest='use_context_reid', action='store_true',
                       help='启用半身衣物上下文辅助匹配（默认开启）')
    parser.add_argument('--no-use-context-reid', dest='use_context_reid', action='store_false',
                       help='禁用半身衣物上下文辅助匹配')
    parser.set_defaults(use_context_reid=cfg.USE_CONTEXT_REID)
    parser.add_argument('--context-face-weight', type=float, default=cfg.CONTEXT_FACE_WEIGHT,
                       help='融合匹配中人脸embedding权重（默认0.8）')
    parser.add_argument('--context-apparel-weight', type=float, default=cfg.CONTEXT_APPAREL_WEIGHT,
                       help='融合匹配中衣物上下文权重（默认0.2）')
    parser.add_argument('--context-hist-bins', type=int, default=cfg.CONTEXT_HIST_BINS,
                       help='衣物HSV直方图bins（默认8）')

    # 半身截图控制
    parser.add_argument('--save-half-body', dest='save_half_body', action='store_true',
                       help='保存半身截图到 output/<video>/half_body（默认开启）')
    parser.add_argument('--no-save-half-body', dest='save_half_body', action='store_false',
                       help='不保存半身截图')
    parser.set_defaults(save_half_body=cfg.SAVE_HALF_BODY_IMAGE)
    parser.add_argument('--half-body-top-expand', type=float, default=cfg.HALF_BODY_TOP_EXPAND,
                       help='半身框向上扩展比例（相对人脸高度，默认0.25）')
    parser.add_argument('--half-body-bottom-expand', type=float, default=cfg.HALF_BODY_BOTTOM_EXPAND,
                       help='半身框向下扩展比例（相对人脸高度，默认2.2）')
    parser.add_argument('--half-body-side-expand', type=float, default=cfg.HALF_BODY_SIDE_EXPAND,
                       help='半身框两侧扩展比例（相对人脸宽度，默认0.8）')

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

    # 全局匹配总开关（控制跨运行历史库匹配 + 跨视频全局聚类）
    parser.add_argument('--enable-global-match', dest='enable_global_match', action='store_true',
                       help='启用全局匹配（加载历史embedding库 + 跨视频全局聚类，默认开启）')
    parser.add_argument('--disable-global-match', dest='enable_global_match', action='store_false',
                       help='禁用全局匹配（不加载历史库，不做跨视频聚类，仅当前运行内去重）')
    parser.set_defaults(enable_global_match=cfg.ENABLE_GLOBAL_MATCH)

    args = parser.parse_args()

    # 全局匹配总开关联动：关闭时自动关闭 reuse_embedding_db 和 global_cluster_refine
    if not getattr(args, 'enable_global_match', True):
        args.reuse_embedding_db = False
        args.global_cluster_refine = False
        logger.info('ℹ️  全局匹配已关闭 (--disable-global-match)：不加载历史库，不做跨视频聚类')

    if args.embedding_db_dir is None:
        args.embedding_db_dir = args.output_dir
    if args.cluster_threshold is None:
        if args.metric == 'euclidean':
            args.cluster_threshold = cfg.CLUSTER_THRESHOLD_EUCLIDEAN
        else:
            args.cluster_threshold = cfg.CLUSTER_THRESHOLD_COSINE
    if args.global_cluster_threshold is None:
        if args.metric == 'euclidean':
            args.global_cluster_threshold = cfg.GLOBAL_CLUSTER_THRESHOLD_EUCLIDEAN
        else:
            args.global_cluster_threshold = cfg.GLOBAL_CLUSTER_THRESHOLD_COSINE

    # 归一化融合权重，避免误配置
    w_face = max(0.0, float(args.context_face_weight))
    w_app = max(0.0, float(args.context_apparel_weight))
    w_sum = w_face + w_app
    if w_sum <= 1e-12:
        args.context_face_weight = 1.0
        args.context_apparel_weight = 0.0
    else:
        args.context_face_weight = w_face / w_sum
        args.context_apparel_weight = w_app / w_sum

    # half_body 模式下，主匹配特征已经来自半身，不再叠加context二次融合
    if str(args.embedding_mode).strip().lower() == 'half_body':
        args.use_context_reid = False
        args.save_half_body = True
    
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
    logger.info(f"  Embedding模式: {args.embedding_mode}")
    logger.info(f"  姿态阈值: yaw={args.yaw_threshold}°, pitch={args.pitch_threshold}°, roll={args.roll_threshold}°")
    logger.info(f"  去重阈值: {args.threshold} ({args.metric})")
    logger.info(f"  聚类精修: {args.cluster_refine} (threshold={args.cluster_threshold}, min_size={args.cluster_min_size})")
    logger.info(
        f"  全局聚类精修: {args.global_cluster_refine} "
        f"(threshold={args.global_cluster_threshold}, min_size={args.global_cluster_min_size})"
    )
    logger.info(
        f"  上下文匹配: {args.use_context_reid} "
        f"(face_w={args.context_face_weight:.2f}, apparel_w={args.context_apparel_weight:.2f}, bins={args.context_hist_bins})"
    )
    logger.info(
        f"  半身截图: {args.save_half_body} "
        f"(top={args.half_body_top_expand}, bottom={args.half_body_bottom_expand}, side={args.half_body_side_expand})"
    )
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

    if getattr(args, 'global_cluster_refine', False):
        merged_count, global_count, file_count = refine_global_persona_ids_with_clustering(
            output_root=args.output_dir,
            metric=args.metric,
            threshold=args.global_cluster_threshold,
            min_cluster_size=args.global_cluster_min_size,
            embedding_mode=args.embedding_mode,
        )
        if merged_count > 0:
            logger.info(
                f"✅ 跨视频全局聚类已完成: 合并 {merged_count} 个跨视频重复persona, "
                f"全局ID数 {global_count}, 处理记录文件 {file_count} 个"
            )
        else:
            logger.info("ℹ️  跨视频全局聚类未发现可合并的跨视频persona")


if __name__ == '__main__':
    main()