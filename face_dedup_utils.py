"""face_dedup_utils.py

人脸去重和检测的核心工具函数库

核心模块：
1. HeadPoseEstimator - 头部姿态估计（yaw/pitch/roll）
2. FaceQualityResult - 人脸质量评估结果
3. Detector - 统一检测器接口（InsightFace/YOLO）
4. Deduper - 人脸去重器
5. 人脸对齐、特征提取等工具函数
"""

from __future__ import annotations
import os
import sys

# 在导入任何 YOLO 和 ONNX 库之前禁用日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow
os.environ['ONNXRUNTIME_LOGS_LEVEL'] = 'FATAL'  # ONNX Runtime
os.environ['YOLO_VERBOSE'] = 'False'  # Ultralytics YOLO
import warnings
warnings.filterwarnings('ignore')

import math
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

# 可选依赖检测
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception:
    INSIGHTFACE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

import config as cfg


# ============================================================================
# 头部姿态估计
# ============================================================================

class HeadPoseEstimator:
    """
    头部姿态估计器
    通过面部关键点计算头部姿态角度（yaw, pitch, roll）
    """
    
    def __init__(self):
        # 3D面部模型点（标准正脸坐标系）
        # 这些点是基于平均人脸模型的3D坐标
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # 鼻尖 (30)
            (0.0, -330.0, -65.0),        # 下巴 (8)
            (-225.0, 170.0, -135.0),     # 左眼左角 (36)
            (225.0, 170.0, -135.0),      # 右眼右角 (45)
            (-150.0, -150.0, -125.0),    # 左嘴角 (48)
            (150.0, -150.0, -125.0)      # 右嘴角 (54)
        ], dtype=np.float64)
        
        # 简化的5点模型（对应 insightface 的 kps）
        # 索引: 0=左眼, 1=右眼, 2=鼻子, 3=左嘴角, 4=右嘴角
        self.model_points_5 = np.array([
            (-30.0, 60.0, -30.0),    # 左眼
            (30.0, 60.0, -30.0),     # 右眼
            (0.0, 0.0, 0.0),         # 鼻子
            (-20.0, -30.0, -20.0),   # 左嘴角
            (20.0, -30.0, -20.0),    # 右嘴角
        ], dtype=np.float64)
    
    def estimate_pose(self, kps: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        估计头部姿态角度
        
        Args:
            kps: 关键点数组，shape=(5,2) 或 (N,2)
            image_shape: 图像尺寸 (height, width)
        
        Returns:
            (yaw, pitch, roll): 角度（度）
        """
        if kps is None or len(kps) < 5:
            return 0.0, 0.0, 0.0
        
        kps = np.asarray(kps, dtype=np.float64)
        
        # 使用几何方法估计姿态
        # 基于眼睛和鼻子的相对位置
        
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        
        # 计算 Yaw（左右转动）
        # 通过鼻子的x坐标相对于眼睛中心的位置来判断
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_width = np.linalg.norm(right_eye - left_eye)
        
        if eye_width < 1e-6:
            return 0.0, 0.0, 0.0
        
        # 鼻子相对于眼睛中心的水平偏移
        nose_offset_x = nose[0] - eye_center_x
        # 估计 yaw 角度（简化公式）
        yaw = np.degrees(np.arctan2(nose_offset_x, eye_width * 1.5))
        
        # 计算 Pitch（上下转动）
        # 通过鼻子的y坐标相对于眼睛中心的位置来判断
        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
        nose_offset_y = nose[1] - eye_center_y
        
        # 正常情况下，鼻子应该在眼睛下方
        # 如果鼻子偏上，说明抬头；偏下说明低头
        expected_nose_y = eye_width * 0.6  # 预期鼻子在眼睛下方约0.6倍眼距
        pitch = np.degrees(np.arctan2(nose_offset_y - expected_nose_y, eye_width))
        
        # 计算 Roll（头部倾斜）
        # 通过两眼的连线角度来判断
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll = np.degrees(np.arctan2(dy, dx))
        
        return yaw, pitch, roll
    
    def estimate_pose_cv2(self, kps: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        使用 OpenCV 的 solvePnP 估计头部姿态（更精确）
        
        Args:
            kps: 关键点数组，shape=(5,2)
            image_shape: 图像尺寸 (height, width)
        
        Returns:
            (yaw, pitch, roll): 角度（度）
        """
        if kps is None or len(kps) < 5:
            return 0.0, 0.0, 0.0
        
        kps = np.asarray(kps, dtype=np.float64)
        h, w = image_shape[:2]
        
        # 相机内参矩阵
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))  # 无畸变
        
        # 2D图像点
        image_points = kps.reshape(-1, 2).astype(np.float64)
        
        # 对应的3D模型点
        model_points = self.model_points_5[:len(image_points)]
        
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return self.estimate_pose(kps, image_shape)
            
            # 将旋转向量转换为旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # 从旋转矩阵提取欧拉角
            sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
            
            if sy < 1e-6:
                pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = 0
            else:
                pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            
            return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)
            
        except Exception as e:
            # 回退到简化方法
            return self.estimate_pose(kps, image_shape)
    
    def is_frontal_face(self, yaw: float, pitch: float, roll: float,
                        yaw_threshold: float = 25.0,
                        pitch_threshold: float = 25.0,
                        roll_threshold: float = 15.0) -> Tuple[bool, float]:
        """
        判断是否为正脸
        
        Args:
            yaw, pitch, roll: 姿态角度
            yaw_threshold: yaw角度阈值（度）
            pitch_threshold: pitch角度阈值（度）
            roll_threshold: roll角度阈值（度）
        
        Returns:
            (is_frontal, pose_score): 是否为正脸，姿态分数(0-1)
        """
        is_frontal = (abs(yaw) <= yaw_threshold and 
                      abs(pitch) <= pitch_threshold and 
                      abs(roll) <= roll_threshold)
        
        # 计算姿态分数（角度越小分数越高）
        yaw_score = max(0, 1 - abs(yaw) / 90.0)
        pitch_score = max(0, 1 - abs(pitch) / 90.0)
        roll_score = max(0, 1 - abs(roll) / 45.0)
        
        pose_score = (yaw_score + pitch_score + roll_score) / 3.0
        
        # 添加详细的角度检查日志
        logger.info(f"   [姿态检查] yaw={yaw:.1f}° (±{yaw_threshold}°), "
                   f"pitch={pitch:.1f}° (±{pitch_threshold}°), "
                   f"roll={roll:.1f}° (±{roll_threshold}°) → "
                   f"{'✓正脸' if is_frontal else '⚠️非正脸'}, 姿态分数={pose_score:.3f}")
        
        return is_frontal, pose_score


# 全局姿态估计器实例
_head_pose_estimator = HeadPoseEstimator()


# ============================================================================
# 改进的质量评估
# ============================================================================

@dataclass
class FaceQualityResult:
    """人脸质量评估结果"""
    is_high_quality: bool
    is_frontal: bool
    quality_score: float
    pose_score: float
    yaw: float
    pitch: float
    roll: float
    reasons: List[str]


def evaluate_face_quality(
    det: Dict,
    image_shape: Tuple[int, int] = None,
    face_image: np.ndarray = None,  # 新增：实际的人脸裁剪图像，用于物理质量检查
    # 姿态阈值
    yaw_threshold: float = 25.0,
    pitch_threshold: float = 25.0,
    roll_threshold: float = 15.0,
    # 质量阈值
    min_eye_distance: float = 20.0,
    min_face_size: int = 50,
    # 清晰度阈值
    min_sharpness: float = 0.1,
    # 是否严格模式
    strict_mode: bool = False,
    # 调试：是否输出详细日志
    debug: bool = False
) -> FaceQualityResult:
    """
    综合评估人脸质量
    
    Args:
        det: 检测结果字典，包含 bbox, kps, confidence 等
        image_shape: 图像尺寸
        face_image: 实际的人脸裁剪图像，用于检查亮度/对比度/清晰度（新增）
        yaw_threshold: yaw角度阈值
        pitch_threshold: pitch角度阈值
        roll_threshold: roll角度阈值
        min_eye_distance: 最小眼距
        min_face_size: 最小人脸尺寸
        min_sharpness: 最小清晰度
        strict_mode: 是否严格模式
        debug: 是否输出详细日志（用于调试筛选过程）
    
    Returns:
        FaceQualityResult: 评估结果
    """
    reasons = []
    scores = {}
    
    # 获取检测信息用于日志
    det_bbox = det.get('bbox', None)
    det_conf = det.get('confidence', 0.0)
    if det_bbox is not None:
        det_w = det_bbox[2] - det_bbox[0]
        det_h = det_bbox[3] - det_bbox[1]
        logger.info(f"[质量评估] 人脸尺寸={det_w:.0f}x{det_h:.0f}px, 置信度={det_conf:.3f}, "
                   f"姿态阈值=(yaw={yaw_threshold}°, pitch={pitch_threshold}°, roll={roll_threshold}°)")
    
    # ============= 新增：物理图像质量检查 =============
    image_quality_score = 1.0
    if face_image is not None and face_image.size > 0:
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # 亮度检查已移除 - 允许任何亮度级别的人脸
            brightness = np.mean(gray)
            
            # 对比度检查已移除 - 允许任何对比度的人脸
            contrast = np.std(gray)
            
            # 清晰度检查（Laplacian方差，推荐>= 100）
            # 修复：使用正确的 ddepth 参数 CV_64F
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = float(np.var(laplacian))
            if sharpness < 50:  # 非常模糊
                reasons.append(f'图像模糊: {sharpness:.1f}（推荐>=100）')
                image_quality_score *= 0.4
            elif sharpness < 100:  # 略显模糊
                reasons.append(f'清晰度偏低: {sharpness:.1f}（推荐>=100）')
                image_quality_score *= 0.8
        except Exception as e:
            logger.debug(f"图像质量检查异常: {e}")
    
    # 1. 检查关键点
    kps = det.get('kps', None)
    
    # 如果没有关键点，无法评估姿态，默认认为是正脸（使用 YOLO ONNX 时这是正常情况）
    if kps is None:
        # 没有关键点的情况：使用基础置信度评估
        confidence = float(det.get('confidence', 0.5))
        # 如果没有关键点且图像质量较差，则降低评分
        final_quality = confidence * image_quality_score
        return FaceQualityResult(
            is_high_quality=final_quality > 0.5,  # 需要通过图像质量检查
            is_frontal=True,  # 默认认为是正脸（因为无法评估）
            quality_score=final_quality,
            pose_score=1.0,  # 满分（无法评估）
            yaw=0.0, pitch=0.0, roll=0.0,
            reasons=reasons if final_quality < 0.5 else []
        )
    
    kps = np.asarray(kps)
    if kps.shape[0] < 3:
        return FaceQualityResult(
            is_high_quality=False,
            is_frontal=False,
            quality_score=0.0,
            pose_score=0.0,
            yaw=0.0, pitch=0.0, roll=0.0,
            reasons=['关键点数量不足'] + reasons
        )
    
    # 2. 检查人脸尺寸与位置（严格限制小人脸与误入镜）
    bbox = det.get('bbox', None)
    if bbox is not None:
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        # 注：已删除人脸占原图百分比的检查（MIN_FACE_TO_IMAGE_RATIO）
        # 该检查会导致视频中的远距人脸被过度过滤
        # 现已改为仅依赖MIN_SAVE_FACE_SIZE进行绝对尺寸检查
        
        # 固定最小人脸尺寸（从 config 获取，防止误入镜的小人脸被识别）
        fixed_min_size = getattr(cfg, 'MIN_SAVE_FACE_SIZE', 100)
        
        logger.info(f"   [尺寸检查] 宽度={face_width}px (最小{fixed_min_size}px), "
                   f"高度={face_height}px (最小{fixed_min_size}px), "
                   f"通过={face_width >= fixed_min_size or face_height >= fixed_min_size}")
        
        if face_width < fixed_min_size and face_height < fixed_min_size:
            reason = f'人脸尺寸过小: {face_width}x{face_height}px（最小{fixed_min_size}px），宽高至少一个需≥{fixed_min_size}px'
            reasons.append(reason)
            logger.warning(f"❌ [尺寸检查] {reason}")
            # 直接返回，不继续处理过小的人脸
            return FaceQualityResult(
                is_high_quality=False,
                is_frontal=False,
                quality_score=0.0,
                pose_score=0.0,
                yaw=0.0, pitch=0.0, roll=0.0,
                reasons=reasons
            )
        
        # 检查宽高比（防止拉伸或压扁的人脸）
        aspect_ratio = face_width / face_height if face_height > 0 else 1.0
        
        # 宽高比检查已移除 - 允许任何宽高比的人脸（包括极端异形脸）
    
    # 3. 检查眼距
    left_eye = kps[0]
    right_eye = kps[1]
    eye_dist = np.linalg.norm(left_eye - right_eye)
    
    if eye_dist < min_eye_distance:
        reason = f'眼距过小: {eye_dist:.1f}px'
        reasons.append(reason)
        if debug:
            logger.debug(f"⚠️  {reason}")
    
    # 4. 估计头部姿态
    if image_shape is None and bbox is not None:
        # 从bbox推断图像尺寸（注意避免 image_shape 被修改后影响上面的百分比计算）
        # 这里仅用于姿态估计，不影响之前的百分比检查
        inferred_image_shape = (bbox[3] + 100, bbox[2] + 100)  # 留一些余量
    else:
        inferred_image_shape = image_shape
    
    if inferred_image_shape is not None:
        yaw, pitch, roll = _head_pose_estimator.estimate_pose(kps, inferred_image_shape)
    else:
        yaw, pitch, roll = _head_pose_estimator.estimate_pose(kps, (200, 200))
    
    # 5. 判断是否为正脸
    is_frontal, pose_score = _head_pose_estimator.is_frontal_face(
        yaw, pitch, roll,
        yaw_threshold=yaw_threshold,
        pitch_threshold=pitch_threshold,
        roll_threshold=roll_threshold
    )
    
    if not is_frontal:
        reason = f'非正脸: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°（记录为参考，但仍保留此脸）'
        reasons.append(reason)
        logger.info(f"   ⚠️ {reason}")
    else:
        logger.info(f"   ✅ 正脸")
    
    # 6. 计算综合质量分数
    # 眼距分数
    eye_dist_score = min(1.0, eye_dist / (min_eye_distance * 2))
    
    # 置信度分数
    confidence = det.get('confidence', 0.5)
    confidence_score = confidence
    
    # 综合分数：包含检测质量和物理图像质量
    quality_score = (pose_score * 0.5 + 
                     eye_dist_score * 0.25 + 
                     confidence_score * 0.25) * image_quality_score
    
    logger.info(f"   [综合分数] pose={pose_score:.3f}*0.5 + eye_dist={eye_dist_score:.3f}*0.25 + conf={confidence_score:.3f}*0.25 = {quality_score:.3f}")
    
    # 7. 判断是否高质量
    # ⚠️改为：不要求必须是正脸，只要没有致命问题就保留
    is_high_quality = len(reasons) == 0
    
    if strict_mode:
        # 严格模式下，要求更高的姿态分数
        is_high_quality = is_high_quality and pose_score > 0.7
    
    return FaceQualityResult(
        is_high_quality=is_high_quality,
        is_frontal=is_frontal,
        quality_score=quality_score,
        pose_score=pose_score,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        reasons=reasons
    )


def is_high_quality_face(
    det: Dict,
    yaw_threshold: float = 25.0,
    pitch_threshold: float = 25.0,
    roll_threshold: float = 15.0,
    min_eye_distance: float = 20.0
) -> bool:
    """
    高质量人脸判断

    Args:
        det: 检测结果字典
        yaw_threshold: yaw角度阈值
        pitch_threshold: pitch角度阈值
        roll_threshold: roll角度阈值
        min_eye_distance: 最小眼距

    Returns:
        bool: 是否为高质量正脸
    """
    result = evaluate_face_quality(
        det,
        yaw_threshold=yaw_threshold,
        pitch_threshold=pitch_threshold,
        roll_threshold=roll_threshold,
        min_eye_distance=min_eye_distance
    )
    return result.is_high_quality


# ============================================================================
# 改进的检测器
# ============================================================================

def _iou(boxA, boxB):
    """计算两个框的 IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return interArea / union


class Detector:
    """
    统一检测器接口：优先使用 insightface，其次 YOLO (.pt > .onnx)
    
    优先级：
    1. InsightFace (最优：有关键点+embedding)
    2. YOLO .pt 模型 (优：可以提取关键点进行pose estimation)
    3. YOLO .onnx 模型 (备选：无关键点，需要额外验证)
    """

    def __init__(self, backend: str = 'auto', device: str = 'cuda', yolo_weights: Optional[str] = None):
        self.device = device
        self.backend = backend
        self.app = None
        self.model = None
        self.face_validator = None  # YOLO ONNX 模式下用于验证真假人脸

        if backend in ['insightface', 'auto'] and INSIGHTFACE_AVAILABLE:
            try:
                # 优先检查本地模型目录
                local_model_path = os.path.abspath(os.path.join('.', 'models', 'insightface', 'buffalo_l'))
                
                # 检查本地模型是否存在（检查关键文件）
                required_files = ['det_10g.onnx', 'w600k_r50.onnx', 'genderage.onnx']
                local_model_ready = os.path.exists(local_model_path)
                
                if local_model_ready:
                    # 验证关键模型文件是否存在
                    model_files = os.listdir(local_model_path) if os.path.isdir(local_model_path) else []
                    has_all_files = all(f in model_files for f in required_files[:1])  # 至少检查主要模型
                    
                    if has_all_files:
                        logger.info(f"📂 发现本地 InsightFace 模型: {local_model_path}")
                        # 直接加载本地模型
                        self.app = FaceAnalysis(name='buffalo_l', root=os.path.dirname(local_model_path))
                    else:
                        logger.warning(f"⚠️  本地模型不完整，尝试使用默认配置")
                        self.app = FaceAnalysis(name='buffalo_l')
                else:
                    logger.info(f"📥 本地模型不存在，尝试使用默认配置（将从网络下载）")
                    self.app = FaceAnalysis(name='buffalo_l')
                
                try:
                    self.app.prepare(ctx_id=0 if device == 'cuda' else -1)
                except Exception as prepare_err:
                    logger.debug(f"GPU 初始化失败，尝试 CPU: {prepare_err}")
                    self.app.prepare(ctx_id=-1)
                
                self.backend = 'insightface'
                logger.info(f"✅ 使用 InsightFace 检测器 (最优，包含关键点和embedding)")
            except Exception as e:
                logger.error(f"❌ InsightFace 初始化失败: {e}")
                logger.debug(f"   详细错误: {type(e).__name__}: {str(e)}")
                self.app = None

        if self.app is None and (backend in ['yolo', 'auto'] and YOLO_AVAILABLE):
            try:
                # 支持 .pt 和 .onnx 格式
                # 优先级：.pt 人脸模型 > .onnx 人脸模型
                yolo_model = yolo_weights
                if yolo_model is None:
                    # 1. 尝试 .pt 人脸专用模型
                    pt_models = [
                        'models/yolo/yolov11n-face.pt',
                        'models/yolo/yolov11s-face.pt',
                        'models/yolo/yolov11m-face.pt',
                        'models/yolo/yolo26n-face.pt',
                        'models/yolo/yolov10n-face.pt',
                        './yolov11n-face.pt',
                        './yolo26n-face.pt',
                        'yolov11n-face.pt',
                    ]
                    
                    for pt_path in pt_models:
                        if os.path.exists(pt_path):
                            logger.info(f"📥 找到 .pt 人脸模型: {pt_path}")
                            yolo_model = pt_path
                            break
                    
                    # 2. 如果没有找到 .pt，尝试 .onnx 人脸专用模型
                    if yolo_model is None:
                        onnx_models = [
                            'models/yolo/yolov11n-face.onnx',
                            'models/yolo/yolo26n-face.onnx',
                            './yolov11n-face.onnx',
                            'yolov11n-face.onnx',
                        ]
                        
                        for onnx_path in onnx_models:
                            if os.path.exists(onnx_path):
                                logger.warning(f"⚠️  未找到 .pt 模型，使用 ONNX: {onnx_path}")
                                logger.warning(f"   建议下载 .pt 模型获得更好的准确性")
                                yolo_model = onnx_path
                                # 初始化人脸验证器（ONNX模式下无关键点）
                                self.face_validator = FaceValidityChecker()
                                break
                    
                    # 3. 如果都没有，使用默认
                    if yolo_model is None:
                        yolo_model = 'yolov11n-face.pt'
                        logger.warning(f"未找到预下载的人脸模型，将尝试从网络加载: {yolo_model}")
                
                # 初始化 YOLO 模型
                if yolo_model.endswith('.pt'):
                    self.model = YOLO(yolo_model)
                    self.backend = 'yolo'
                    logger.info(f"✅ 使用 YOLO .pt 检测器 (优：可提取关键点)")
                    logger.info(f"   模型: {yolo_model}")
                elif yolo_model.endswith('.onnx'):
                    self.model = yolo_model  # 存储路径，在 predict 时使用
                    self.backend = 'yolo'
                    logger.info(f"⚠️  使用 YOLO .onnx 检测器 (备选：无关键点)")
                    logger.info(f"   模型: {yolo_model}")
                    # ONNX 模式下初始化验证器
                    if self.face_validator is None:
                        self.face_validator = FaceValidityChecker()
                else:
                    # 尝试初始化，让 YOLO 自动判断格式
                    self.model = YOLO(yolo_model)
                    self.backend = 'yolo'
                    logger.info(f"使用 YOLO 检测器 (backend={self.backend}, model={yolo_model})")
            except Exception as e:
                logger.warning(f"YOLO 初始化失败: {e}")
                self.model = None

        if self.app is None and self.model is None:
            self.backend = 'none'
            logger.error("没有可用的检测器! (Haar 已被禁用，仅支持 insightface/yolo)")

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """检测人脸并返回检测结果列表"""
        if self.backend == 'insightface' and self.app is not None:
            try:
                faces = self.app.get(frame)
                out = []
                for f in faces:
                    bbox = f.bbox.astype(int).tolist() if hasattr(f, 'bbox') else None
                    kps = np.array(f.kps) if hasattr(f, 'kps') and f.kps is not None else None
                    emb = np.array(f.embedding) if hasattr(f, 'embedding') and f.embedding is not None else None
                    score = float(getattr(f, 'det_score', 1.0))
                    if score < conf_threshold:
                        continue
                    out.append({
                        'bbox': (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                        'kps': kps, 
                        'confidence': score, 
                        'embedding': emb
                    })
                return out
            except Exception as e:
                logger.warning(f"InsightFace 检测失败: {e}")

        if self.backend == 'yolo' and self.model is not None:
            try:
                # 设置推理设备
                yolo_device = 0 if self.device == 'cuda' else 'cpu'
                
                # 暂时关闭 stderr (FD 2) 以抑制 ONNX Runtime C++ 日志
                import sys
                old_stderr = sys.stderr.fileno() if hasattr(sys.stderr, 'fileno') else None
                if old_stderr is not None:
                    old_fd = os.dup(old_stderr)
                    devnull_fd = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull_fd, old_stderr)
                
                try:
                    # 处理两种情况：YOLO 对象或 ONNX 路径字符串
                    if isinstance(self.model, str):
                        # ONNX 格式：直接用 YOLO().predict() 调用，传递 device
                        results = YOLO(self.model).predict(frame, device=yolo_device, verbose=False, conf=0.5, task='detect')
                    else:
                        # PyTorch 格式：使用缓存的 YOLO 对象
                        results = self.model.predict(frame, device=yolo_device, verbose=False, conf=0.5, task='detect')
                finally:
                    # 恢复 stderr
                    if old_stderr is not None:
                        os.dup2(old_fd, old_stderr)
                        os.close(old_fd)
                        os.close(devnull_fd)
                
                r = results[0]
                boxes = getattr(r, 'boxes', None)
                if boxes is None or boxes.data is None:
                    return []
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy().reshape(-1)
                out = []
                for j in range(xyxy.shape[0]):
                    conf = float(confs[j])
                    if conf < conf_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, xyxy[j])
                    out.append({
                        'bbox': (x1, y1, x2, y2), 
                        'kps': None, 
                        'confidence': conf, 
                        'embedding': None
                    })
                return out
            except Exception as e:
                logger.warning(f"YOLO 检测失败: {e}")

        return []


# ============================================================================
# 人脸对齐与美观保存
# ============================================================================

def align_face(img: np.ndarray, kps: np.ndarray, output_size: Tuple[int, int] = None) -> np.ndarray:
    """根据左右眼关键点做仿射对齐，output_size 从 config 获取默认值"""
    if output_size is None:
        output_size = cfg.ALIGNED_FACE_SIZE

    if kps is None or len(kps) < 2:
        return cv2.resize(img, output_size)

    kps = np.asarray(kps)
    left_eye = np.array(kps[0], dtype=float)
    right_eye = np.array(kps[1], dtype=float)
    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return cv2.resize(img, output_size)

    desired_left = cfg.ALIGNMENT_PARAMS.get('desired_left', (0.35, 0.35))
    desired_right_x = cfg.ALIGNMENT_PARAMS.get('desired_right_x', 0.65)
    desired_dist = (desired_right_x - desired_left[0]) * output_size[0]
    scale = desired_dist / dist

    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
    tX = output_size[0] * 0.5
    tY = output_size[1] * desired_left[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    aligned = cv2.warpAffine(img, M, output_size, flags=cv2.INTER_CUBIC)
    return aligned


def save_face_pretty(img: np.ndarray, bbox: Tuple[int, int, int, int], 
                    kps: Optional[np.ndarray] = None, output_size: int = 224) -> np.ndarray:
    """
    以美观的方式保存人脸，避免变形
    
    当提供了关键点时，自动旋转人脸使其水平
    添加适量的padding以展现面部周围环境，更自然美观
    
    Args:
        img: 输入图像
        bbox: 人脸bbox (x1, y1, x2, y2)
        kps: 可选的关键点，用于自动旋转校正
        output_size: 输出图像大小（保持宽高比）
    
    Returns:
        美观的人脸图像，保持人脸宽高比，避免压缩变形
    """
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    
    # 计算padding（在原始bbox基础上扩展）
    pad_y = max(int(h * 0.15), 10)  # 上下各扩展15%或10像素
    pad_x = max(int(w * 0.1), 5)    # 左右各扩展10%或5像素
    
    # 计算扩展后的裁剪区域
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(img.shape[1], x2 + pad_x)
    crop_y2 = min(img.shape[0], y2 + pad_y)
    
    face_crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    
    # 如果提供了关键点，进行旋转校正
    if kps is not None and len(kps) >= 2:
        try:
            # 调整关键点坐标到cropped图像坐标系
            kps_adjusted = np.asarray(kps, dtype=float)
            kps_adjusted[:, 0] -= crop_x1
            kps_adjusted[:, 1] -= crop_y1
            
            # 计算左右眼角度
            left_eye = kps_adjusted[0]
            right_eye = kps_adjusted[1]
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = math.degrees(math.atan2(dy, dx))
            
            # 旋转人脸使其水平
            h_crop, w_crop = face_crop.shape[:2]
            center = (w_crop // 2, h_crop // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            face_crop = cv2.warpAffine(face_crop, M, (w_crop, w_crop), flags=cv2.INTER_CUBIC)
        except Exception as e:
            logger.debug(f"人脸旋转校正失败: {e}")
    
    # 保持宽高比缩放（不强行压缩）
    h_face, w_face = face_crop.shape[:2]
    aspect_ratio = w_face / h_face
    
    if aspect_ratio > 1:
        # 宽度更大，以宽度为基准
        new_w = output_size
        new_h = int(output_size / aspect_ratio)
    else:
        # 高度更大或相等，以高度为基准
        new_h = output_size
        new_w = int(output_size * aspect_ratio)
    
    # 使用立方插值保证质量
    face_resized = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 使用白色背景（或柔和的灰色）铺底
    background = np.ones((output_size, output_size, 3), dtype=np.uint8) * 245  # 浅灰色背景
    
    # 将缩放后的人脸居中放在背景上
    y_offset = (output_size - new_h) // 2
    x_offset = (output_size - new_w) // 2
    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = face_resized
    
    return background


class FaceSaveQualityValidator:
    """保存后的人脸质量检验器"""
    
    def __init__(self, 
                 min_face_width: int = None,
                 min_face_height: int = None,
                 max_aspect_ratio: float = None,  # 最大宽高比，避免过度变形
                 min_brightness: float = None,      # 最小亮度，避免太暗
                 max_brightness: float = None,     # 最大亮度，避免过曝
                 min_contrast: float = None):     # 最小对比度
        # 从 config 中获取默认值（如果未显式传入）
        fv = getattr(cfg, 'FACE_SAVE_VALIDATOR', {})
        self.min_face_width = min_face_width if min_face_width is not None else fv.get('min_face_width', 50)
        self.min_face_height = min_face_height if min_face_height is not None else fv.get('min_face_height', 50)
        self.max_aspect_ratio = max_aspect_ratio if max_aspect_ratio is not None else fv.get('max_aspect_ratio', 1.5)
        self.min_brightness = min_brightness if min_brightness is not None else fv.get('min_brightness', 50)
        self.max_brightness = max_brightness if max_brightness is not None else fv.get('max_brightness', 200)
        self.min_contrast = min_contrast if min_contrast is not None else fv.get('min_contrast', 0.15)
        self.logger = logging.getLogger(__name__)
    
    def validate(self, img: np.ndarray, img_path: str = "") -> Tuple[bool, str]:
        """
        验证保存的人脸图像质量
        
        Args:
            img: 人脸图像
            img_path: 图像路径（用于日志）
            
        Returns:
            (is_valid, reason)
        """
        h, w = img.shape[:2]
        
        self.logger.info(f"[保存验证] 图像尺寸={w}x{h}px, min_width={self.min_face_width}px, min_height={self.min_face_height}px")
        
        # 1. 检查尺寸（只要长或宽其中一个满足即可）
        if w < self.min_face_width and h < self.min_face_height:
            reason = f"人脸尺寸过小: {w}x{h} (最小宽: {self.min_face_width}px 或 最小高: {self.min_face_height}px)"
            self.logger.warning(f"❌ [保存验证-尺寸] {reason}")
            return False, reason
        else:
            self.logger.info(f"   ✅ [尺寸] 通过 (宽={w}>={self.min_face_width} or 高={h}>={self.min_face_height})")
        
        # 2. 宽高比检查已移除 - 允许任何宽高比的人脸（包括极端异形脸）
        
        # 3. 检查亮度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        brightness = gray.mean()
        contrast = gray.std()
        # 亮度检查已移除 - 允许任何亮度级别的人脸
        self.logger.info(f"   [亮度/对比度] 亮度={brightness:.1f}, 对比度={contrast:.1f} (检查已移除)")
        
        # 4. 检查对比度（避免严重模糊）
        # 对比度检查已移除 - 允许任何对比度的人脸
        
        # 5. 检查人脸区域（避免全是背景）
        # 检查非背景像素的比例
        if len(img.shape) == 3:
            # 浅灰色背景的像素范围
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            non_bg_mask = (gray_img < 240) | (gray_img > 250)  # 不是背景色的像素
            if len(img.shape) == 3:
                non_bg_ratio = np.mean(non_bg_mask)
            else:
                non_bg_ratio = np.mean(non_bg_mask)
            
            self.logger.info(f"   [背景] 非背景像素占比={non_bg_ratio*100:.1f}%")
            if non_bg_ratio < 0.3:
                reason = f"人脸内容过少: 仅{non_bg_ratio*100:.1f}% (最少30%)"
                self.logger.warning(f"❌ [保存验证-背景] {reason}")
                return False, reason
        
        # 如果所有检验都通过
        self.logger.info(f"✅ [保存验证] 所有检查通过: 尺寸={w}x{h}, 亮度={brightness:.1f}, 对比度={contrast:.1f}")
        return True, "通过"


# ============================================================================
# 人脸有效性检查 (严格分类，避免误检)
# ============================================================================

class FaceValidityChecker:
    """
    严格的人脸有效性检查器
    
    目的：排除鼠标、手机等误检，只保留真实人脸
    方法：多层验证
    1. 色彩空间特性检查 (皮肤色调)
    2. 熵值检查 (防止单色物体)
    3. 边缘特性检查 (人脸有典型的边缘分布)
    4. 尺寸比例检查 (人脸有合理的宽高比)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _get_skin_tone_ratio(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        计算人脸区域中的皮肤色调像素比例
        
        皮肤色调范围（RGB和HSV空间）：
        - R: 95-220,  G: 40-200,  B: 20-170
        - H: 0-50 或 330-360 (红黄范围)
        """
        x1, y1, x2, y2 = bbox
        face_crop = img[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return 0.0
        
        # 转换到HSV空间
        try:
            if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
            else:
                return 0.0
            
            # 计算HSV范围内的像素（皮肤色调）
            # H: 0-50 (红黄) 或 330-360 (红)
            # S: 10-150 (饱和度不能太低或太高)
            # V: 50-255 (亮度)
            mask1 = cv2.inRange(hsv, (0, 10, 50), (50, 150, 255))      # 黄-红范围
            mask2 = cv2.inRange(hsv, (170, 10, 50), (180, 150, 255))   # 纯红范围
            mask = cv2.bitwise_or(mask1, mask2)
            
            skin_pixels = cv2.countNonZero(mask)
            total_pixels = face_crop.shape[0] * face_crop.shape[1]
            skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0.0
            
            return min(1.0, skin_ratio)
        except Exception as e:
            self.logger.debug(f"皮肤色调检查失败: {e}")
            return 0.5  # 无法判断时返回中立值
    
    def _get_entropy(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        计算人脸区域的信息熵
        
        纯色物体（如手机屏幕）的熵很低
        真实人脸有较高的熵（纹理丰富）
        """
        x1, y1, x2, y2 = bbox
        face_crop = img[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return 0.0
        
        try:
            # 转换为灰度
            if len(face_crop.shape) == 3:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_crop
            
            # 计算直方图
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / gray.size
            
            # 计算熵
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return min(entropy / 8.0, 1.0)  # 归一化到0-1
        except Exception as e:
            self.logger.debug(f"熵计算失败: {e}")
            return 0.5
    
    def _get_edge_density(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        计算边缘密度
        
        人脸有明显的边缘特征（眼睛、嘴巴、脸部轮廓）
        手机、鼠标等物体边缘分布不同
        """
        x1, y1, x2, y2 = bbox
        
        # 确保坐标在图像范围内
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face_crop = img[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return 0.0
        
        try:
            # 转换为灰度
            if len(face_crop.shape) == 3:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_crop
            
            # Canny边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 计算边缘密度
            edge_pixels = cv2.countNonZero(edges)
            total_pixels = gray.shape[0] * gray.shape[1]
            edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # 人脸边缘密度通常在 0.05-0.30 之间
            # 太低（< 0.02）可能是单色物体
            # 太高（> 0.40）可能是纹理图案或文本
            return edge_density
        except Exception as e:
            self.logger.debug(f"边缘密度计算失败: {e}")
            return 0.15  # 无法判断时返回中立值
    
    def _get_aspect_ratio(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        获取边界框的宽高比
        
        人脸的宽高比有典型范围
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        if h == 0:
            return 1.0
        
        ratio = w / h
        # 人脸宽高比通常在 0.7-1.3 之间
        return min(ratio, 1 / ratio)  # 始终返回 <= 1 的值
    
    def is_valid_face(self, img: np.ndarray, bbox: Tuple[int, int, int, int],
                     confidence: float = 0.5) -> Tuple[bool, Dict[str, float]]:
        """
        进行严格的人脸有效性检查
        
        Args:
            img: 输入图像
            bbox: 问题区域的边界框 (x1, y1, x2, y2)
            confidence: 检测置信度
            
        Returns:
            (is_valid, scores_dict)
            - is_valid: 是否是有效的人脸
            - scores_dict: 各项评分详情
        """
        scores = {}
        
        # 1. 置信度检查
        scores['confidence'] = min(1.0, confidence)
        confidence_valid = confidence > 0.3
        
        # 2. 皮肤色调检查
        skin_ratio = self._get_skin_tone_ratio(img, bbox)
        scores['skin_ratio'] = skin_ratio
        # 人脸中至少 15% 的像素应该是皮肤色
        skin_valid = skin_ratio > 0.15
        
        # 3. 熵检查
        entropy = self._get_entropy(img, bbox)
        scores['entropy'] = entropy
        # 熵不能太低（单色物体）或太高（噪声）
        entropy_valid = 0.3 < entropy < 0.9
        
        # 4. 边缘密度检查
        edge_density = self._get_edge_density(img, bbox)
        scores['edge_density'] = edge_density
        # 人脸的边缘密度应该在 0.05-0.35 之间
        edge_valid = 0.05 < edge_density < 0.35
        
        # 5. 宽高比检查
        aspect_ratio = self._get_aspect_ratio(bbox)
        scores['aspect_ratio'] = aspect_ratio
        # 人脸的宽高比应该在 0.65-1.0 之间（接近圆形）
        aspect_valid = aspect_ratio > 0.65
        
        # 综合判断
        # 至少要满足以下条件：
        # - 置信度不能太低
        # - 皮肤色比例要合理
        # - 熵（纹理）不能太极端
        # - 边缘密度在人脸的合理范围内
        # - 宽高比接近人脸
        
        is_valid = (
            confidence_valid and       # 必须有基本的置信度
            (skin_valid or edge_valid) and  # 至少有皮肤色或合理的边缘
            entropy_valid and           # 纹理应该合理
            aspect_valid                # 宽高比应该接近人脸
        )
        
        return is_valid, scores
    
    def classify_detection(self, img: np.ndarray, bbox: Tuple[int, int, int, int],
                          confidence: float = 0.5) -> Tuple[str, str, Dict]:
        """
        对检测结果进行分类
        
        Returns:
            (category, reason, scores_dict)
            - category: 'face' (人脸) / 'object' (物体) / 'uncertain' (不确定)
            - reason: 分类原因
            - scores_dict: 各项评分
        """
        is_valid, scores = self.is_valid_face(img, bbox, confidence)
        
        if is_valid:
            # 进一步细分为"高信心"和"低信心"
            confidence_score = scores['confidence']
            skin_score = scores['skin_ratio']
            entropy_score = scores['entropy']
            
            if confidence_score > 0.7 and skin_score > 0.25 and entropy_score > 0.4:
                return 'face', '高置信度人脸', scores
            else:
                return 'face', '低置信度人脸（但通过检查）', scores
        else:
            # 分析失败原因
            reasons = []
            if not scores.get('confidence', 0.5) > 0.3:
                reasons.append(f"置信度过低({scores['confidence']:.2f})")
            if not (scores.get('skin_ratio', 0) > 0.15):
                reasons.append(f"皮肤色比例过低({scores['skin_ratio']:.2f})")
            if not (0.3 < scores.get('entropy', 0.5) < 0.9):
                reasons.append(f"纹理异常({scores['entropy']:.2f})")
            if not (0.05 < scores.get('edge_density', 0.15) < 0.35):
                reasons.append(f"边缘密度异常({scores['edge_density']:.2f})")
            if not (scores.get('aspect_ratio', 0.8) > 0.65):
                reasons.append(f"宽高比异常({scores['aspect_ratio']:.2f})")
            
            reason = " + ".join(reasons) if reasons else "综合评分不符合人脸特征"
            return 'object', reason, scores


# ============================================================================
# 轻量检测驱动追踪器 (Detection -> Tracker)
# ============================================================================

class DetectionTracker:
    """
    基于检测结果的轻量跟踪器（ByteTrack-like 行为的替代方案）+
    采用“检测框中心距离 + 大小差异”作为匹配代价，避免使用 IoU。

    用法:
        tracker = DetectionTracker(max_age=30, max_center_cost=0.7)
        tids = tracker.update(detections, frame_idx)

    detections: list of dict, 每项包含 'bbox'=(x1,y1,x2,y2) 和 'confidence'
    返回: tids: 与 detections 等长的 track id 列表（int）
    """

    def __init__(self, max_age: int = 30, max_center_cost: float = 0.7,
                 center_weight: float = 1.0, size_weight: float = 0.5,
                 min_confidence: float = 0.0):
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 1
        self.max_age = max_age
        self.max_center_cost = max_center_cost
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.min_confidence = min_confidence

    def _bbox_props(self, bbox: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        diag = math.hypot(w, h)
        area = w * h
        return cx, cy, diag, area

    def update(self, detections: List[Dict], frame_idx: int) -> List[Optional[int]]:
        """将本帧的检测结果分配到已有track或创建新track。

        detections: list of dict with keys 'bbox' and 'confidence'
        frame_idx: 当前帧编号，用于 track 老化和删除
        返回: tids 列表，与 detections 索引对应
        """
        assigned_tids: List[Optional[int]] = [None] * len(detections)

        # 1) 快速创建（无历史track）
        if len(self.tracks) == 0:
            for i, det in enumerate(detections):
                conf = float(det.get('confidence', 0.0))
                if conf < self.min_confidence:
                    continue
                bbox = tuple(map(int, det['bbox']))
                cx, cy, diag, area = self._bbox_props(bbox)
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': bbox,
                    'center': (cx, cy),
                    'diag': diag,
                    'area': area,
                    'last_update': frame_idx,
                    'age': 0,
                    'hits': 1,
                }
                assigned_tids[i] = tid
            return assigned_tids

        # 2) 匹配已有track（按置信度从高到低匹配检测）
        unmatched_tracks = set(self.tracks.keys())
        order = sorted(range(len(detections)), key=lambda i: detections[i].get('confidence', 0.0), reverse=True)

        for i in order:
            det = detections[i]
            conf = float(det.get('confidence', 0.0))
            if conf < self.min_confidence:
                continue
            bbox = tuple(map(int, det['bbox']))
            cx, cy, diag, area = self._bbox_props(bbox)

            best_tid = None
            best_cost = float('inf')

            for tid in list(unmatched_tracks):
                tr = self.tracks.get(tid)
                if tr is None:
                    continue
                tcx, tcy = tr['center']
                tdiag = tr['diag']

                center_dist = math.hypot(cx - tcx, cy - tcy)
                scale = max(diag, tdiag, 1.0)
                center_cost = center_dist / scale
                size_cost = abs(diag - tdiag) / scale

                cost = self.center_weight * center_cost + self.size_weight * size_cost
                if cost < best_cost:
                    best_cost = cost
                    best_tid = tid

            # 如果最优代价足够小则匹配，否则新建track
            if best_tid is not None and best_cost <= self.max_center_cost:
                assigned_tids[i] = best_tid
                unmatched_tracks.discard(best_tid)
                # 更新 track 信息
                self.tracks[best_tid].update({
                    'bbox': bbox,
                    'center': (cx, cy),
                    'diag': diag,
                    'area': area,
                    'last_update': frame_idx,
                })
                self.tracks[best_tid]['age'] = 0
                self.tracks[best_tid]['hits'] = self.tracks[best_tid].get('hits', 0) + 1
            else:
                # 创建新 track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': bbox,
                    'center': (cx, cy),
                    'diag': diag,
                    'area': area,
                    'last_update': frame_idx,
                    'age': 0,
                    'hits': 1,
                }
                assigned_tids[i] = tid

        # 3) 老化并清理过期 track
        to_delete = []
        for tid, tr in list(self.tracks.items()):
            if tr.get('last_update') != frame_idx:
                tr['age'] = tr.get('age', 0) + 1
            if tr.get('age', 0) > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return assigned_tids


# ============================================================================
# 改进的 Embedding 提取
# ============================================================================

class SimpleEmbedder:
    """提供 embedding 的回退实现"""

    def __init__(self):
        pass

    @staticmethod
    def get_embedding_from_detection(det: Dict, face_crop: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        emb = det.get('embedding', None)
        if emb is not None:
            v = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(v)
            return v / (norm + 1e-12)

        if face_crop is None:
            return None
        try:
            hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            norm = np.linalg.norm(hist)
            return hist / (norm + 1e-12)
        except Exception:
            return None


# ============================================================================
# 改进的去重器
# ============================================================================

class Deduper:
    """
    改进的基于最近邻的去重器
    
    改进点：
    1. 支持基于姿态的去重
    2. 支持渐进式阈值
    3. 支持聚类合并
    """

    def __init__(self, metric: str = None, threshold: float = None):
        """
        Args:
            metric: 距离度量 ('cosine' 或 'euclidean')
            threshold: 相似度阈值（cosine: 0.5-0.7, euclidean: 0.6-1.0）
        """
        self.metric = metric if metric is not None else cfg.SIMILARITY_METRIC
        self.threshold = float(threshold if threshold is not None else cfg.DEDUP_THRESHOLD)
        self.embeddings: List[np.ndarray] = []
        self.ids: List[int] = []
        self.metadata: List[Dict] = []  # 存储额外信息（如姿态）
        self._next_id = 1

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    @staticmethod
    def _safe_l2_normalize(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if arr.size == 0 or norm <= 0:
            return None
        return arr / norm

    def find_match(self, emb: np.ndarray, metadata: Dict = None, debug: bool = False) -> Optional[int]:
        """
        查找匹配的人脸ID
        
        Args:
            emb: 人脸特征向量
            metadata: 额外信息（如姿态角度）
            debug: 是否输出详细匹配日志
        
        Returns:
            匹配的ID，如果没有匹配则返回 None
        """
        if emb is None or len(self.embeddings) == 0:
            return None
        
        emb_orig = np.asarray(emb, dtype=np.float32)
        emb_orig_norm = np.linalg.norm(emb_orig)
        emb = emb / (emb_orig_norm + 1e-12)

        # 可选：融合半身衣物上下文特征（HSV直方图）
        query_ctx = self._safe_l2_normalize((metadata or {}).get('context_hist'))
        use_context = query_ctx is not None
        face_weight = float((metadata or {}).get('face_weight', getattr(cfg, 'CONTEXT_FACE_WEIGHT', 0.8)))
        context_weight = float((metadata or {}).get('context_weight', getattr(cfg, 'CONTEXT_APPAREL_WEIGHT', 0.2)))
        total_weight = face_weight + context_weight
        if total_weight <= 1e-12:
            face_weight, context_weight = 1.0, 0.0
        else:
            face_weight /= total_weight
            context_weight /= total_weight
        
        if self.metric == 'cosine':
            sims = [float(np.dot(e, emb)) for e in self.embeddings]

            if use_context:
                fused_sims = []
                for i, face_sim in enumerate(sims):
                    cand_ctx = self._safe_l2_normalize((self.metadata[i] or {}).get('context_hist'))
                    if cand_ctx is not None and cand_ctx.shape == query_ctx.shape:
                        ctx_sim = self._cosine_sim(cand_ctx, query_ctx)
                        fused_sims.append(face_weight * face_sim + context_weight * ctx_sim)
                    else:
                        # 缺失上下文时回退到纯人脸分数
                        fused_sims.append(face_sim)
                sims_to_use = fused_sims
            else:
                sims_to_use = sims

            best_idx = int(np.argmax(sims_to_use))
            
            if debug:
                # 显示前3个候选人脸的相似度（用于调试）
                top_3_indices = sorted(range(len(sims_to_use)), key=lambda i: sims_to_use[i], reverse=True)[:3]
                top_3_info = " | ".join([f"ID={self.ids[i]:05d}(sim={sims_to_use[i]:.4f})" for i in top_3_indices])
                logger.info(f"[去重候选] 前3个候选: {top_3_info} | 输入emb_dim={emb_orig.shape[0]} | norm={emb_orig_norm:.6f}")
            
            if sims_to_use[best_idx] >= self.threshold:
                matched_id = self.ids[best_idx]
                logger.info(f"🔗 去重匹配: ID={matched_id:05d} | sim={sims_to_use[best_idx]:.4f} | 输入emb_dim={emb_orig.shape[0]}")
                return matched_id
            return None
        else:
            dists = [float(np.linalg.norm(e - emb)) for e in self.embeddings]

            if use_context:
                fused_dists = []
                for i, face_dist in enumerate(dists):
                    cand_ctx = self._safe_l2_normalize((self.metadata[i] or {}).get('context_hist'))
                    if cand_ctx is not None and cand_ctx.shape == query_ctx.shape:
                        ctx_dist = float(np.linalg.norm(cand_ctx - query_ctx))
                        fused_dists.append(face_weight * face_dist + context_weight * ctx_dist)
                    else:
                        fused_dists.append(face_dist)
                dists_to_use = fused_dists
            else:
                dists_to_use = dists

            best_idx = int(np.argmin(dists_to_use))
            
            if debug:
                # 显示前3个候选人脸的距离（用于调试）
                top_3_indices = sorted(range(len(dists_to_use)), key=lambda i: dists_to_use[i])[:3]
                top_3_info = " | ".join([f"ID={self.ids[i]:05d}(dist={dists_to_use[i]:.4f})" for i in top_3_indices])
                logger.info(f"[去重候选] 前3个候选: {top_3_info} | 输入emb_dim={emb_orig.shape[0]} | norm={emb_orig_norm:.6f}")
            
            if dists_to_use[best_idx] <= self.threshold:
                matched_id = self.ids[best_idx]
                logger.info(f"🔗 去重匹配: ID={matched_id:05d} | dist={dists_to_use[best_idx]:.4f} | 输入emb_dim={emb_orig.shape[0]}")
                return matched_id
            return None

    def add(self, emb: np.ndarray, metadata: Dict = None) -> int:
        """
        添加新的人脸特征
        
        Args:
            emb: 人脸特征向量
            metadata: 额外信息
        
        Returns:
            新分配的ID
        """
        if emb is None:
            raise ValueError('Cannot add empty embedding')
        
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        pid = self._next_id
        self._next_id += 1
        self.embeddings.append(emb)
        self.ids.append(pid)
        self.metadata.append(metadata or {})
        
        # 记录新添加的人脸，包含维度和统计信息以便调试
        logger.info(f"✨ 新建人脸ID: {pid:05d} | emb_dim={emb.shape[0]} | L2_norm={np.linalg.norm(emb):.6f} | "
                   f"mean={np.mean(emb):.6f} | std={np.std(emb):.6f}")
        
        return pid

    def get_all_embeddings(self) -> np.ndarray:
        """获取所有特征向量"""
        if len(self.embeddings) == 0:
            return np.array([])
        return np.stack(self.embeddings, axis=0)


# ============================================================================
# 跟踪历史与连续重复过滤
# ============================================================================

class TrackHistory:
    """
    维护跟踪历史，用于检测连续重复人脸（同一 track_id 多帧连续出现）
    
    功能：
    1. 记录每个 track_id 的出现frame和质量分数
    2. 检查是否为连续重复（同一track在相邻帧出现）
    3. 判断是否应该保存（仅保留该track中质量最好的帧）
    """
    
    def __init__(self, max_age: int = 10):
        """
        Args:
            max_age: track 没有更新时的最大存活帧数
        """
        self.tracks: Dict[int, Dict] = {}  # track_id -> {last_frame, quality, best_quality, best_frame}
        self.max_age = max_age
    
    def update(self, track_id: int, frame_idx: int, quality_score: float = 1.0) -> bool:
        """
        更新 track 记录，返回是否应该保存此帧
        
        Args:
            track_id: 跟踪ID
            frame_idx: 当前帧号
            quality_score: 人脸质量分数（0-1）
        
        Returns:
            True=应该保存此帧, False=跳过（连续重复或质量差）
        """
        if track_id not in self.tracks:
            # 첫 출현
            self.tracks[track_id] = {
                'last_frame': frame_idx,
                'quality': quality_score,
                'best_quality': quality_score,
                'best_frame': frame_idx,
                'first_saved': False
            }
            logger.debug(f"📍 新建 track {track_id} | frame {frame_idx} | 质量 {quality_score:.4f}")
            return True
        
        track = self.tracks[track_id]
        
        # 检查是否连续出现（frame_idx == last_frame + 1）
        is_continuous = (frame_idx == track['last_frame'] + 1)
        
        if is_continuous:
            # 连续出现，检查质量
            if quality_score > track['best_quality']:
                # 比当前最好的质量更好，更新最好记录
                logger.debug(f"🔄 连续 track {track_id} | frame {frame_idx} → {frame_idx-1} | "
                           f"质量提升 {track['best_quality']:.4f} → {quality_score:.4f} | 更新为最好帧")
                track['best_quality'] = quality_score
                track['best_frame'] = frame_idx
                track['quality'] = quality_score
            else:
                logger.debug(f"⏭️  连续重复 track {track_id} | frame {frame_idx} | "
                           f"质量 {quality_score:.4f} ≤ 已保存最好质量 {track['best_quality']:.4f} | 跳过")
            
            track['last_frame'] = frame_idx
            return False  # 连续出现时跳过（等待 track 结束后再保存最好的）
        
        else:
            # 不连续（出现间隔），说明同一 track 再次出现（可能经历了遮挡或短暂消失）
            # 重置为新的出现
            track['last_frame'] = frame_idx
            track['quality'] = quality_score
            track['best_quality'] = quality_score
            track['best_frame'] = frame_idx
            logger.debug(f"🔁 track {track_id} 再次出现 | 间隔 {frame_idx - track['last_frame']} 帧 | 质量 {quality_score:.4f}")
            return True  # 重新出现时保存
    
    def get_best_quality(self, track_id: int) -> Tuple[float, int]:
        """获取 track 的最好质量和对应 frame"""
        if track_id in self.tracks:
            t = self.tracks[track_id]
            return t['best_quality'], t['best_frame']
        return 0.0, -1
    
    def is_continuous_duplicate(self, track_id: int, frame_idx: int) -> bool:
        """检查是否为连续重复（简化版：只检查是否刚刚出现过）"""
        if track_id not in self.tracks:
            return False
        
        track = self.tracks[track_id]
        # 如果上一次出现正好是前一帧，则是连续重复
        return track['last_frame'] == frame_idx - 1
    
    def cleanup(self, current_frame: int):
        """清理过期的 track（超过 max_age 帧没有出现）"""
        to_remove = []
        for track_id, track in list(self.tracks.items()):
            age = current_frame - track['last_frame']
            if age > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        return len(to_remove)



