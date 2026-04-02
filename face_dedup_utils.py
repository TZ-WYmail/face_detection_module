"""face_dedup_utils.py
工具函数与轻量跟踪/检测/对齐/embedding 回退实现

目标：为高质量的人脸去重流水线提供可插拔的检测、对齐、embedding 提取与轨迹聚合能力，
在缺失 heavy 依赖时优雅回退（不会抛异常）。
"""
from __future__ import annotations
import os
import math
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

# 可选依赖检测（不会在 import 时失败）
try:
    from insightface import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception:
    INSIGHTFACE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


def _iou(boxA, boxB):
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


class _SimpleTrack:
    def __init__(self, tid: int, tlbr: Tuple[int, int, int, int], score: float):
        self.track_id = int(tid)
        self.tlbr = [int(v) for v in tlbr]
        self.score = float(score)
        self.time_since_update = 0


class IoUTracker:
    """轻量 IoU 跟踪器：基于 IoU 的贪婪匹配，适合短时轨迹聚合与去重回退。

    参数：
      - iou_threshold: 匹配阈值
      - max_age: 轨迹允许丢失的帧数，超过移除
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.tracks: Dict[int, _SimpleTrack] = {}
        self._next_id = 1

    def update(self, dets: List[List[float]], frame=None) -> List[_SimpleTrack]:
        """
        dets: list of [x1,y1,x2,y2,score]
        返回当前活动轨迹对象列表
        """
        matched_det_idx = set()
        matched_track_ids = set()

        det_boxes = [d[:4] for d in dets]
        det_scores = [d[4] if len(d) > 4 else 0.0 for d in dets]

        track_ids = list(self.tracks.keys())
        if track_ids and det_boxes:
            iou_matrix = []
            for tid in track_ids:
                tr = self.tracks[tid]
                row = [_iou(tr.tlbr, db) for db in det_boxes]
                iou_matrix.append(row)

            used_rows = set()
            used_cols = set()
            while True:
                best_val = 0.0
                best_r = -1
                best_c = -1
                for r, row in enumerate(iou_matrix):
                    if r in used_rows:
                        continue
                    for c, val in enumerate(row):
                        if c in used_cols:
                            continue
                        if val > best_val:
                            best_val = val
                            best_r, best_c = r, c
                if best_val < self.iou_threshold or best_r == -1:
                    break
                tid = track_ids[best_r]
                matched_det_idx.add(best_c)
                matched_track_ids.add(tid)
                self.tracks[tid].tlbr = [int(v) for v in det_boxes[best_c]]
                self.tracks[tid].score = float(det_scores[best_c])
                self.tracks[tid].time_since_update = 0
                used_rows.add(best_r)
                used_cols.add(best_c)

        for d_idx, box in enumerate(det_boxes):
            if d_idx in matched_det_idx:
                continue
            tid = self._next_id
            self._next_id += 1
            tr = _SimpleTrack(tid, box, det_scores[d_idx])
            tr.time_since_update = 0
            self.tracks[tid] = tr

        remove_ids = []
        for tid, tr in list(self.tracks.items()):
            if tid in matched_track_ids:
                continue
            tr.time_since_update += 1
            if tr.time_since_update > self.max_age:
                remove_ids.append(tid)
        for tid in remove_ids:
            self.tracks.pop(tid, None)

        return list(self.tracks.values())


class Detector:
    """统一检测器接口：优先使用 insightface，其次 ultralytics YOLO，最后回退到 OpenCV Haar。

    detect(frame) -> List[dict] 每个 dict 包含：
      - 'bbox': (x1,y1,x2,y2)
      - 'kps': np.ndarray(5,2) 或 None
      - 'confidence': float
      - 'embedding': np.ndarray 或 None (如果检测器本身能提供)
    """

    def __init__(self, backend: str = 'auto', device: str = 'cuda', yolo_weights: Optional[str] = None):
        self.device = device
        self.backend = backend
        self.app = None
        self.model = None
        self.haar = None

        if backend in ['insightface', 'auto'] and INSIGHTFACE_AVAILABLE:
            try:
                self.app = FaceAnalysis(name='buffalo_l')
                # prepare will download models if missing; keep it best-effort
                try:
                    self.app.prepare(ctx_id=0)
                except Exception:
                    # 如果 GPU/提供者不可用，仍允许后续调用在 CPU 上运行
                    self.app.prepare(ctx_id=-1)
                self.backend = 'insightface'
            except Exception:
                self.app = None

        if self.app is None and (backend in ['yolo', 'auto'] and YOLO_AVAILABLE):
            try:
                self.model = YOLO(yolo_weights or 'yolov11n-face.pt')
                self.backend = 'yolo'
            except Exception:
                self.model = None

        if self.app is None and self.model is None:
            # Haar 回退
            haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(haar_path):
                self.haar = cv2.CascadeClassifier(haar_path)
                self.backend = 'haar'
            else:
                self.haar = None
                self.backend = 'none'

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
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
                    out.append({'bbox': (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                                'kps': kps, 'confidence': score, 'embedding': emb})
                return out
            except Exception:
                # 回退到下一种实现
                pass

        if self.backend == 'yolo' and self.model is not None:
            try:
                results = self.model(frame)
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
                    out.append({'bbox': (x1, y1, x2, y2), 'kps': None, 'confidence': conf, 'embedding': None})
                return out
            except Exception:
                pass

        if self.backend == 'haar' and self.haar is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                out = []
                for (x, y, w, h) in rects:
                    out.append({'bbox': (int(x), int(y), int(x + w), int(y + h)), 'kps': None, 'confidence': 1.0, 'embedding': None})
                return out
            except Exception:
                pass

        return []


def align_face(img: np.ndarray, kps: np.ndarray, output_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """根据左右眼关键点做仿射对齐并输出固定大小的 crop（若 kps 缺失则返回原裁剪）。
    kps: array-like 5x2 or similar，索引0=left_eye,1=right_eye,2=nose,..."""
    if kps is None or kps.shape[0] < 2:
        return cv2.resize(img, output_size)

    left_eye = np.array(kps[0], dtype=float)
    right_eye = np.array(kps[1], dtype=float)
    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return cv2.resize(img, output_size)

    desired_left = (0.35, 0.35)
    desired_right_x = 1.0 - desired_left[0]
    desired_dist = (desired_right_x - desired_left[0]) * output_size[0]
    scale = desired_dist / dist

    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
    tX = output_size[0] * 0.5
    tY = output_size[1] * desired_left[1]
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    aligned = cv2.warpAffine(img, M, output_size, flags=cv2.INTER_CUBIC)
    return aligned


def is_high_quality_face(det: Dict, min_eye_distance: float = 20.0, nose_offset_thresh: float = 0.4) -> bool:
    """基于关键点做快速质量过滤（按 Design.md 建议）。
    若缺失关键点，则返回 False（严格策略，可放宽）。
    """
    kps = det.get('kps', None)
    if kps is None:
        return False
    kps = np.asarray(kps)
    if kps.shape[0] < 3:
        return False
    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]
    eye_dist = np.linalg.norm(left_eye - right_eye)
    if eye_dist < min_eye_distance:
        return False
    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    nose_offset_ratio = abs(nose[0] - eye_center_x) / (eye_dist + 1e-6)
    if nose_offset_ratio > nose_offset_thresh:
        return False
    return True


class SimpleEmbedder:
    """提供 embedding 的回退实现：如果 detection 已包含 embedding 则直接返回；
    否则用 HSV 直方图作为轻量特征向量（归一化）。
    """

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


class Deduper:
    """简单的基于最近邻的去重器（保存 normalized embeddings）。
    metric: 'cosine' or 'euclidean'（默认 cosine），threshold 为相似阈值。
    """

    def __init__(self, metric: str = 'cosine', threshold: float = 0.6):
        self.metric = metric
        self.threshold = float(threshold)
        self.embeddings: List[np.ndarray] = []
        self.ids: List[int] = []
        self._next_id = 1

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def find_match(self, emb: np.ndarray) -> Optional[int]:
        if emb is None or len(self.embeddings) == 0:
            return None
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        if self.metric == 'cosine':
            sims = [float(np.dot(e, emb)) for e in self.embeddings]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= self.threshold:
                return self.ids[best_idx]
            return None
        else:
            dists = [float(np.linalg.norm(e - emb)) for e in self.embeddings]
            best_idx = int(np.argmin(dists))
            if dists[best_idx] <= self.threshold:
                return self.ids[best_idx]
            return None

    def add(self, emb: np.ndarray) -> int:
        if emb is None:
            raise ValueError('Cannot add empty embedding')
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        pid = self._next_id
        self._next_id += 1
        self.embeddings.append(emb)
        self.ids.append(pid)
        return pid
