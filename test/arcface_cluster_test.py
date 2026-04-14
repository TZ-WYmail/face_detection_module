#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ArcFace 人脸聚类测试脚本
========================
使用 InsightFace w600k_r50 (ArcFace) 模型提取人脸 embedding，
通过 Union-Find 聚类判断图片中的人脸是否属于同一人。

与 cluster_test.py (OSNet ReID) 的区别:
  - cluster_test.py    → OSNet 看半身/衣着，适合区分不同穿搭的人
  - 本脚本             → ArcFace 看人脸五官，适合精确判断是否同一人

输入源可以是:
  - face_ref/    (纯人脸裁剪图，推荐)
  - half_body/   (半身图，ArcFace 会自动检测其中的人脸)
  - 任意含人脸的图片文件夹

用法示例:
    # 单文件夹聚类 (推荐输入 face_ref/)
    python test/arcface_cluster_test.py ./output/video-3/face_ref/ --threshold 0.45 --verbose

    # 跨文件夹全局聚类
    python test/arcface_cluster_test.py ./output/ --global --threshold 0.45 --verbose

    # 指定 InsightFace 模型目录
    python test/arcface_cluster_test.py ./output/video-3/face_ref/ --model-root ./models/insightface
"""

import argparse
import csv
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """配置日志。"""
    level = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 支持的图片扩展名
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# ArcFace 人脸特征提取器
# ---------------------------------------------------------------------------
class ArcFaceFeatureExtractor:
    """
    基于 InsightFace w600k_r50 (ArcFace) 的人脸特征提取器。
    使用 insightface.app.FaceAnalysis 加载模型，输出 512 维 L2 归一化 embedding。

    ArcFace 的 cosine similarity 在同一人上通常可达 0.85~0.95+，
    不同人一般在 0.3~0.6 之间，区分度远高于 OSNet ReID 或 HSV 直方图。
    """

    def __init__(self, model_root: str = None, model_name: str = 'buffalo_l', device: str = 'cuda'):
        self.model_root = model_root  # insightface 模型根目录 (包含 buffalo_l/)
        self.model_name = model_name
        self.device = device
        self.app = None
        self._loaded = False
        self._embed_dim = 512

    def _lazy_load(self):
        """延迟加载 InsightFace 模型。"""
        if self._loaded:
            return
        try:
            from insightface.app import FaceAnalysis

            kwargs = {}
            if self.model_root:
                kwargs['root'] = os.path.abspath(self.model_root)
                kwargs['name'] = self.model_name
            else:
                kwargs['name'] = self.model_name

            self.app = FaceAnalysis(**kwargs)

            ctx_id = 0 if (self.device == 'cuda') else -1
            try:
                self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            except Exception:
                self.app.prepare(ctx_id=-1, det_size=(640, 640))

            self._loaded = True
            logger.info(
                f'✅ ArcFace 模型已加载 (model={self.model_name}, '
                f'device={"GPU" if ctx_id == 0 else "CPU"})'
            )
        except ImportError:
            logger.error('⚠️ insightface 未安装，无法使用 ArcFace。请安装: pip install insightface')
        except Exception as e:
            logger.error(f'⚠️ ArcFace 模型加载失败: {e}')

    def is_available(self) -> bool:
        """检查模型是否可用。"""
        self._lazy_load()
        return self.app is not None

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        从图片中检测人脸并提取 ArcFace embedding。

        如果图片中有多张人脸，返回面积最大的人脸的 embedding。
        如果未检测到人脸，返回 None。

        Args:
            image: BGR 格式图片 (np.ndarray)

        Returns:
            L2 归一化的 512 维 float32 向量，失败返回 None。
        """
        if image is None or image.size == 0:
            return None

        self._lazy_load()
        if self.app is None:
            return None

        try:
            faces = self.app.get(image)
            if not faces:
                logger.debug('  未检测到人脸')
                return None

            # 选择面积最大的人脸
            bbox_areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
            best_idx = int(np.argmax(bbox_areas))
            face = faces[best_idx]

            # ArcFace embedding 已经是 L2 归一化的
            emb = face.embedding.astype(np.float32)
            if emb is None or emb.size == 0:
                return None

            # 再次确保归一化
            norm = np.linalg.norm(emb)
            if norm <= 0:
                return None
            return emb / norm

        except Exception as e:
            logger.debug(f'ArcFace 特征提取失败: {e}')
            return None

    @property
    def embed_dim(self) -> int:
        return self._embed_dim


# ---------------------------------------------------------------------------
# 核心算法：Union-Find 聚类
# ---------------------------------------------------------------------------

class UnionFind:
    """并查集（路径压缩 + 按秩合并）。"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def compute_similarity(v1: np.ndarray, v2: np.ndarray, metric: str) -> float:
    """计算两个归一化向量之间的相似度分数。"""
    if metric == "euclidean":
        return float(np.linalg.norm(v1 - v2))
    else:  # cosine (dot product of normalized vectors)
        return float(np.dot(v1, v2))


def union_find_clustering(
    keys: List[str],
    proto_mat: np.ndarray,
    metric: str = "cosine",
    threshold: float = 0.45,
    min_cluster_size: int = 2,
    log_details: bool = False,
) -> Tuple[Dict[str, int], Dict[int, List[str]], List[Dict[str, Any]]]:
    """
    Union-Find 聚类。

    Returns:
        key_to_cluster_id:  每个 key → 簇编号 (1-based)
        cluster_id_to_keys: 簇编号 → [key list]
        pair_details:       所有 pair 的比较详情
    """
    n = len(keys)
    uf = UnionFind(n)
    pair_details: List[Dict[str, Any]] = []

    for i in range(n):
        vi = proto_mat[i]
        for j in range(i + 1, n):
            vj = proto_mat[j]
            score = compute_similarity(vi, vj, metric)

            if metric == "euclidean":
                is_match = score <= threshold
            else:
                is_match = score >= threshold

            detail = {
                "key_i": keys[i],
                "key_j": keys[j],
                "score": round(score, 6),
                "metric": metric,
                "threshold": threshold,
                "matched": is_match,
            }
            pair_details.append(detail)

            if is_match:
                uf.union(i, j)

            if log_details:
                status = "MERGE" if is_match else "SKIP "
                logger.debug(
                    f"  [{status}] {keys[i]} <-> {keys[j]} | "
                    f"score={score:.6f} threshold={threshold} ({metric})"
                )

    # 收集簇
    root_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        root_to_indices[uf.find(idx)].append(idx)

    cluster_id_to_keys: Dict[int, List[str]] = {}
    key_to_cluster_id: Dict[str, int] = {}
    sorted_roots = sorted(root_to_indices.keys())

    for cid, root in enumerate(sorted_roots, start=1):
        members = sorted(keys[i] for i in root_to_indices[root])
        cluster_id_to_keys[cid] = members
        for k in members:
            key_to_cluster_id[k] = cid

    return key_to_cluster_id, cluster_id_to_keys, pair_details


# ---------------------------------------------------------------------------
# 单文件夹聚类
# ---------------------------------------------------------------------------

def cluster_images_in_folder(
    image_dir: str,
    output_dir: str,
    extractor: ArcFaceFeatureExtractor,
    metric: str = "cosine",
    threshold: float = 0.45,
    min_cluster_size: int = 2,
    log_details: bool = False,
) -> Dict[str, Any]:
    """
    对指定文件夹下的所有图片用 ArcFace 提取人脸特征并聚类。
    """
    logger.info("=" * 70)
    logger.info("ArcFace 人脸聚类任务开始")
    logger.info(f"  输入目录  : {os.path.abspath(image_dir)}")
    logger.info(f"  输出目录  : {os.path.abspath(output_dir)}")
    logger.info(f"  相似度度量: {metric}")
    logger.info(f"  聚类阈值  : {threshold}")
    logger.info(f"  最小簇大小: {min_cluster_size}")
    logger.info("=" * 70)

    # 1. 扫描图片
    image_paths: List[str] = []
    for fname in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            image_paths.append(os.path.join(image_dir, fname))

    if not image_paths:
        logger.warning("未找到任何图片文件，退出。")
        return {"status": "no_images", "image_count": 0}

    logger.info(f"[步骤1] 扫描到 {len(image_paths)} 张图片")

    # 2. 提取 ArcFace 人脸特征
    logger.info("[步骤2] 提取 ArcFace 人脸特征 ...")
    keys: List[str] = []
    embeddings: List[np.ndarray] = []
    failed: List[str] = []
    no_face: List[str] = []

    t0 = time.time()
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is None:
                failed.append(img_path)
                continue

            emb = extractor.extract(img)
            if emb is None:
                no_face.append(os.path.basename(img_path))
                logger.debug(f"  未检测到人脸: {os.path.basename(img_path)}")
                continue

            keys.append(os.path.basename(img_path))
            embeddings.append(emb)
            logger.debug(f"  OK: {os.path.basename(img_path)} | emb_dim={emb.size}")
        except Exception as e:
            failed.append(img_path)
            logger.debug(f"  异常: {img_path} | {e}")

    t_extract = time.time() - t0
    logger.info(
        f"[步骤2] 完成: 成功={len(keys)}, 无人脸={len(no_face)}, 读取失败={len(failed)}, "
        f"耗时={t_extract:.2f}s"
    )

    if len(keys) < 2:
        logger.warning(f"有效图片不足 2 张 (仅 {len(keys)} 张)，无法聚类。")
        return {"status": "too_few", "valid_count": len(keys)}

    # 3. 原型向量（每个 key 归一化）
    proto_vecs = []
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        if norm <= 0:
            proto_vecs.append(emb)
        else:
            proto_vecs.append((emb / norm).astype(np.float32))
    proto_mat = np.stack(proto_vecs, axis=0)
    logger.info(f"[步骤3] 原型矩阵: shape={proto_mat.shape}")

    # 4. Union-Find 聚类
    logger.info(f"[步骤4] 执行 Union-Find 聚类 (threshold={threshold}, metric={metric}) ...")
    t1 = time.time()
    key_to_cid, cid_to_keys, pair_details = union_find_clustering(
        keys, proto_mat, metric, threshold, min_cluster_size, log_details
    )
    t_cluster = time.time() - t1
    logger.info(f"[步骤4] 聚类完成，耗时={t_cluster:.2f}s")

    # 5. 统计
    total_clusters = len(cid_to_keys)
    valid_clusters = sum(
        1 for members in cid_to_keys.values() if len(members) >= min_cluster_size
    )
    singleton_clusters = sum(
        1 for members in cid_to_keys.values() if len(members) == 1
    )
    max_cluster_size = max(len(m) for m in cid_to_keys.values())
    merged_pairs = sum(1 for d in pair_details if d["matched"])

    logger.info("-" * 70)
    logger.info("[步骤5] 聚类结果统计:")
    logger.info(f"  总簇数       : {total_clusters}")
    logger.info(f"  有效簇 (≥{min_cluster_size}): {valid_clusters}")
    logger.info(f"  单例簇       : {singleton_clusters}")
    logger.info(f"  最大簇大小   : {max_cluster_size}")
    logger.info(f"  比较 pair 数 : {len(pair_details)}")
    logger.info(f"  合并 pair 数 : {merged_pairs}/{len(pair_details)}")
    logger.info("-" * 70)

    # 6. 各簇详情
    logger.info("[步骤6] 各簇详情:")
    for cid in sorted(cid_to_keys.keys()):
        members = cid_to_keys[cid]
        status = "✓ 有效" if len(members) >= min_cluster_size else "✗ 单例"
        logger.info(f"  Cluster #{cid}: {len(members)} 张 [{status}]")
        for k in sorted(members):
            logger.info(f"    - {k}")

    # 7. 边缘 case
    logger.info("[步骤7] 边缘 case (score 接近阈值):")
    if metric == "cosine":
        edge_pairs = [d for d in pair_details if abs(d["score"] - threshold) < 0.05]
    else:
        edge_pairs = [d for d in pair_details if abs(d["score"] - threshold) < threshold * 0.1]

    if edge_pairs:
        for d in sorted(edge_pairs, key=lambda x: abs(x["score"] - threshold)):
            status = "MERGE" if d["matched"] else "SKIP "
            logger.info(
                f"  [{status}] {d['key_i']} <-> {d['key_j']} | "
                f"score={d['score']:.6f} (阈值={d['threshold']}, "
                f"差值={abs(d['score'] - d['threshold']):.6f})"
            )
    else:
        logger.info("  (无)")

    # 8. 输出文件
    os.makedirs(output_dir, exist_ok=True)

    # 8a. 聚类映射 CSV
    mapping_path = os.path.join(output_dir, "arcface_cluster_mapping.csv")
    with open(mapping_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "cluster_id", "cluster_size"])
        for k in keys:
            cid = key_to_cid[k]
            writer.writerow([k, cid, len(cid_to_keys[cid])])
    logger.info(f"[输出] 聚类映射: {mapping_path}")

    # 8b. Pair 详情 CSV
    pairs_path = os.path.join(output_dir, "arcface_pair_details.csv")
    with open(pairs_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["key_i", "key_j", "score", "metric", "threshold", "matched"]
        )
        writer.writeheader()
        writer.writerows(pair_details)
    logger.info(f"[输出] Pair 详情: {pairs_path}")

    # 8c. 簇摘要 CSV
    summary_path = os.path.join(output_dir, "arcface_cluster_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "size", "is_valid", "filenames"])
        for cid in sorted(cid_to_keys.keys()):
            members = cid_to_keys[cid]
            writer.writerow([
                cid, len(members), len(members) >= min_cluster_size, "; ".join(sorted(members)),
            ])
    logger.info(f"[输出] 簇摘要: {summary_path}")

    # 8d. 按簇复制图片
    images_out_dir = os.path.join(output_dir, "clustered_images")
    if os.path.exists(images_out_dir):
        shutil.rmtree(images_out_dir)
    os.makedirs(images_out_dir, exist_ok=True)

    for cid, members in cid_to_keys.items():
        cluster_dir = os.path.join(images_out_dir, f"cluster_{cid:03d}")
        os.makedirs(cluster_dir, exist_ok=True)
        for fname in members:
            src = os.path.join(image_dir, fname)
            dst = os.path.join(cluster_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
    logger.info(f"[输出] 按簇分类图片: {images_out_dir}/")

    logger.info("=" * 70)
    logger.info("ArcFace 人脸聚类任务完成!")
    logger.info("=" * 70)

    return {
        "status": "ok",
        "image_count": len(image_paths),
        "valid_count": len(keys),
        "no_face_count": len(no_face),
        "failed_count": len(failed),
        "total_clusters": total_clusters,
        "valid_clusters": valid_clusters,
        "singleton_clusters": singleton_clusters,
        "max_cluster_size": max_cluster_size,
        "total_pairs": len(pair_details),
        "merged_pairs": merged_pairs,
        "time_extract_sec": round(t_extract, 3),
        "time_cluster_sec": round(t_cluster, 3),
    }


# ---------------------------------------------------------------------------
# 跨文件夹全局聚类
# ---------------------------------------------------------------------------

def cluster_global(
    output_root: str,
    extractor: ArcFaceFeatureExtractor,
    metric: str = "cosine",
    threshold: float = 0.45,
    min_cluster_size: int = 2,
    log_details: bool = False,
) -> Dict[str, Any]:
    """
    跨文件夹全局 ArcFace 聚类。
    扫描 output_root 下所有子目录（如 video-1/, video-2/），对每张图片
    用 ArcFace 提取人脸 embedding 后统一聚类。
    """
    logger.info("=" * 70)
    logger.info("ArcFace 跨文件夹全局聚类任务开始")
    logger.info(f"  根目录    : {os.path.abspath(output_root)}")
    logger.info(f"  相似度度量: {metric}")
    logger.info(f"  聚类阈值  : {threshold}")
    logger.info(f"  最小簇大小: {min_cluster_size}")
    logger.info("=" * 70)

    # 1. 扫描子目录
    subdirs: Dict[str, List[str]] = {}
    for entry in sorted(os.listdir(output_root)):
        entry_path = os.path.join(output_root, entry)
        if not os.path.isdir(entry_path):
            continue
        # 跳过非图片目录
        images = []
        for fname in sorted(os.listdir(entry_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(os.path.join(entry_path, fname))
        if images:
            subdirs[entry] = images

    if not subdirs:
        logger.warning("未找到包含图片的子目录，退出。")
        return {"status": "no_dirs"}

    logger.info(f"[步骤1] 扫描到 {len(subdirs)} 个子目录:")
    for d, imgs in subdirs.items():
        logger.info(f"  {d}: {len(imgs)} 张图片")

    # 2. 提取特征
    logger.info("[步骤2] 提取 ArcFace 人脸特征 ...")
    source_keys: List[str] = []
    source_embeddings: List[np.ndarray] = []
    source_meta: Dict[str, Dict[str, str]] = {}
    no_face_total = 0
    failed_total = 0
    t0 = time.time()

    for subdir, img_paths in subdirs.items():
        for img_path in img_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    failed_total += 1
                    continue

                emb = extractor.extract(img)
                if emb is None:
                    no_face_total += 1
                    continue

                fname = os.path.basename(img_path)
                src_key = f"{subdir}/{fname}"
                source_keys.append(src_key)
                source_embeddings.append(emb)
                source_meta[src_key] = {"subdir": subdir, "filename": fname}
            except Exception:
                failed_total += 1

    t_extract = time.time() - t0
    logger.info(
        f"[步骤2] 完成: 有效={len(source_keys)}, 无人脸={no_face_total}, "
        f"失败={failed_total}, 耗时={t_extract:.2f}s"
    )

    if len(source_keys) < 2:
        logger.warning(f"有效图片不足 2 张，无法聚类。")
        return {"status": "too_few", "valid_count": len(source_keys)}

    # 3. 原型向量
    proto_vecs = []
    for emb in source_embeddings:
        v = emb / (np.linalg.norm(emb) + 1e-12)
        proto_vecs.append(v.astype(np.float32))
    proto_mat = np.stack(proto_vecs, axis=0)
    logger.info(f"[步骤3] 原型矩阵 shape={proto_mat.shape}")

    # 4. 聚类
    logger.info(f"[步骤4] 全局 Union-Find 聚类 ...")
    t1 = time.time()
    key_to_cid, cid_to_keys, pair_details = union_find_clustering(
        source_keys, proto_mat, metric, threshold, min_cluster_size, log_details
    )
    t_cluster = time.time() - t1
    logger.info(f"[步骤4] 聚类完成, 耗时={t_cluster:.2f}s")

    # 5. 统计
    total_clusters = len(cid_to_keys)
    valid_clusters = sum(1 for m in cid_to_keys.values() if len(m) >= min_cluster_size)
    merged_pairs = sum(1 for d in pair_details if d["matched"])

    logger.info("-" * 70)
    logger.info("[步骤5] 全局聚类结果:")
    logger.info(f"  总簇数       : {total_clusters}")
    logger.info(f"  有效簇 (≥{min_cluster_size}): {valid_clusters}")
    logger.info(f"  合并 pair 数 : {merged_pairs}/{len(pair_details)}")
    logger.info("-" * 70)

    logger.info("[步骤6] 各簇详情:")
    for cid in sorted(cid_to_keys.keys()):
        members = cid_to_keys[cid]
        dirs_involved = sorted(set(
            source_meta[k]["subdir"] for k in members if k in source_meta
        ))
        status = "✓ 跨目录" if len(dirs_involved) > 1 else "  单目录"
        logger.info(f"  Cluster #{cid}: {len(members)} 张 [{status}] 来源: {dirs_involved}")
        for k in sorted(members):
            meta = source_meta.get(k, {})
            logger.info(f"    - {meta.get('filename', k)} (来自 {meta.get('subdir', '?')})")

    # 7. 写输出
    out_dir = os.path.join(output_root, "arcface_global_cluster_result")
    os.makedirs(out_dir, exist_ok=True)

    mapping_path = os.path.join(out_dir, "arcface_global_cluster_mapping.csv")
    with open(mapping_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_key", "subdir", "filename", "global_cluster_id", "cluster_size"])
        for k in source_keys:
            cid = key_to_cid[k]
            meta = source_meta.get(k, {})
            writer.writerow([k, meta.get("subdir", ""), meta.get("filename", ""), cid, len(cid_to_keys[cid])])
    logger.info(f"[输出] 全局映射: {mapping_path}")

    pairs_path = os.path.join(out_dir, "arcface_global_pair_details.csv")
    with open(pairs_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["key_i", "key_j", "score", "metric", "threshold", "matched"]
        )
        writer.writeheader()
        writer.writerows(pair_details)
    logger.info(f"[输出] Pair 详情: {pairs_path}")

    logger.info("=" * 70)
    logger.info("ArcFace 全局聚类任务完成!")
    logger.info("=" * 70)

    return {
        "status": "ok",
        "dir_count": len(subdirs),
        "valid_count": len(source_keys),
        "no_face_count": no_face_total,
        "failed_count": failed_total,
        "total_clusters": total_clusters,
        "valid_clusters": valid_clusters,
        "merged_pairs": merged_pairs,
        "time_extract_sec": round(t_extract, 3),
        "time_cluster_sec": round(t_cluster, 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ArcFace 人脸聚类测试工具（使用 InsightFace w600k_r50 提取人脸 embedding）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单文件夹聚类 (推荐输入 face_ref/)
  python test/arcface_cluster_test.py ./output/video-3/face_ref/ --threshold 0.45 --verbose

  # 跨文件夹全局聚类
  python test/arcface_cluster_test.py ./output/ --global --threshold 0.45 --verbose

  # 欧氏距离模式
  python test/arcface_cluster_test.py ./output/video-3/face_ref/ --metric euclidean --threshold 1.0

  # 指定模型目录
  python test/arcface_cluster_test.py ./output/video-3/face_ref/ --model-root ./models/insightface
        """,
    )

    parser.add_argument(
        "input_path", type=str,
        help="图片文件夹路径（单文件夹模式）或根目录路径（全局模式）",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="输出目录（默认: <input_path>_arcface_result）",
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.45,
        help="聚类阈值（cosine: 0~1, 越高越严格; euclidean: 越低越严格。推荐: cosine 0.45~0.55）",
    )
    parser.add_argument(
        "--metric", "-m", type=str, choices=["cosine", "euclidean"], default="cosine",
        help="相似度度量方式（默认: cosine）",
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=2,
        help="最小簇大小，低于此数的簇标记为无效（默认: 2）",
    )
    parser.add_argument(
        "--global-mode", "-g", action="store_true", dest="global_mode",
        help="启用跨文件夹全局聚类模式（扫描 input_path 下所有子目录）",
    )
    parser.add_argument(
        "--model-root", type=str, default="models/insightface",
        help="InsightFace 模型根目录路径（默认: models/insightface）",
    )
    parser.add_argument(
        "--model-name", type=str, default="buffalo_l",
        help="InsightFace 模型名称（默认: buffalo_l）",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="推理设备（默认: cuda）",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="输出详细日志（包含每个 pair 的比较分数）",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="将日志同时写入指定文件",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input_path):
        print(f"错误: 路径不存在: {args.input_path}")
        sys.exit(1)

    # 配置日志
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    if args.log_file:
        logger.info(f"日志文件: {os.path.abspath(args.log_file)}")

    # 初始化 ArcFace 提取器
    extractor = ArcFaceFeatureExtractor(
        model_root=args.model_root,
        model_name=args.model_name,
        device=args.device,
    )
    if not extractor.is_available():
        logger.error("ArcFace 模型不可用，请检查 insightface 是否安装且模型文件存在。")
        sys.exit(1)

    # 执行聚类
    if args.global_mode:
        result = cluster_global(
            output_root=args.input_path,
            extractor=extractor,
            metric=args.metric,
            threshold=args.threshold,
            min_cluster_size=args.min_cluster_size,
            log_details=args.verbose,
        )
    else:
        output_dir = args.output_dir or f"{args.input_path}_arcface_result"
        result = cluster_images_in_folder(
            image_dir=args.input_path,
            output_dir=output_dir,
            extractor=extractor,
            metric=args.metric,
            threshold=args.threshold,
            min_cluster_size=args.min_cluster_size,
            log_details=args.verbose,
        )

    # 汇总
    if result.get("status") == "ok":
        logger.info(f"\n📊 汇总: 特征提取 {result['time_extract_sec']}s, "
                     f"聚类 {result['time_cluster_sec']}s, "
                     f"有效簇 {result['valid_clusters']} / 总簇 {result['total_clusters']}, "
                     f"合并 {result['merged_pairs']}/{result['total_pairs']} pairs")


if __name__ == "__main__":
    main()
