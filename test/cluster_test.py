#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立的半身图片聚类测试脚本
==========================
使用与主流程 (face_dedup_pipeline.py) 相同的 Union-Find 聚类逻辑，
对指定文件夹下的半身图片进行离线聚类，并输出详细日志。

支持两种特征提取模式：
  - reid  : OSNet-x0.25 深度特征（推荐，区分度最高）
  - hsv   : HSV 颜色直方图（传统方法，不依赖模型）

用法示例:
    # 使用 ReID 深度特征聚类（推荐）
    python test/cluster_test.py ./output/video-3/half_body/ --feature reid --threshold 0.45

    # 使用 HSV 直方图聚类
    python test/cluster_test.py ./output/video-3/half_body/ --feature hsv --threshold 0.45

    # 跨视频全局聚类
    python test/cluster_test.py ./output/ --global --feature reid --threshold 0.45

    # 指定模型路径
    python test/cluster_test.py ./output/video-3/half_body/ --feature reid --reid-model ./models/reid/osnet_x0_25_msmt17.pt
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
    """配置详细日志，同时输出到控制台和文件。"""
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
# ReID 深度特征提取器（与主流程完全一致）
# ---------------------------------------------------------------------------


class ReIDFeatureExtractor:
    """
    基于 OSNet-x0.25 (MSMT17 预训练) 的 ReID 特征提取器。
    输出 512 维 L2 归一化 embedding。
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model = None
        self.device = device
        self.model_path = model_path
        self._loaded = False

    def _lazy_load(self):
        if self._loaded:
            return
        try:
            import torch
            if not os.path.exists(self.model_path):
                logger.error(
                    f'ReID 模型文件不存在: {self.model_path}。'
                    f'请先运行 python setup_project.py 下载模型。'
                )
                return
            if torch.cuda.is_available() and self.device != 'cpu':
                self.model = torch.jit.load(self.model_path, map_location='cuda').eval()
                self.device = 'cuda'
                logger.info(f'ReID 模型已加载 (GPU): {self.model_path}')
            else:
                self.model = torch.jit.load(self.model_path, map_location='cpu').eval()
                self.device = 'cpu'
                logger.info(f'ReID 模型已加载 (CPU): {self.model_path}')
            self._loaded = True
        except ImportError:
            logger.error('PyTorch 未安装，无法使用 ReID 特征。请安装 PyTorch。')
        except Exception as e:
            logger.error(f'ReID 模型加载失败: {e}')

    def is_available(self) -> bool:
        self._lazy_load()
        return self.model is not None

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None
        self._lazy_load()
        if self.model is None:
            return None
        try:
            import torch
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
            if self.device == 'cuda':
                tensor = tensor.cuda()
            with torch.no_grad():
                feat = self.model(tensor)
            feat = feat.cpu().numpy().flatten().astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm <= 0 or feat.size == 0:
                return None
            return feat / norm
        except Exception as e:
            logger.debug(f'ReID 特征提取失败: {e}')
            return None


# ---------------------------------------------------------------------------
# 统一特征提取入口
# ---------------------------------------------------------------------------


def extract_features(
    image: np.ndarray,
    feature_mode: str = 'reid',
    hist_bins: int = 8,
    reid_extractor: Optional[ReIDFeatureExtractor] = None,
) -> Optional[np.ndarray]:
    """
    统一的特征提取入口。
    feature_mode='reid': 优先使用 OSNet 深度特征，不可用时回退到 HSV。
    feature_mode='hsv' : 仅使用 HSV 直方图。
    """
    if feature_mode == 'reid' and reid_extractor is not None and reid_extractor.is_available():
        emb = reid_extractor.extract(image)
        if emb is not None:
            return emb
        logger.debug('ReID 提取返回空，回退到 HSV 直方图')

    # 回退: HSV 直方图
    return extract_apparel_context_hist(image, hist_bins=hist_bins)


# ---------------------------------------------------------------------------
# 核心算法：HSV 直方图提取（与主流程完全一致）
# ---------------------------------------------------------------------------


def extract_apparel_context_hist(image: np.ndarray, hist_bins: int = 8) -> Optional[np.ndarray]:
    """
    从半身图下半部分提取 HSV 直方图作为衣物上下文特征。
    逻辑与 face_dedup_pipeline.extract_apparel_context_hist() 完全一致。
    """
    if image is None or image.size == 0:
        return None
    try:
        hh, ww = image.shape[:2]
        if hh < 10 or ww < 10:
            return None

        start_y = int(hh * 0.45)
        apparel_region = image[start_y:, :]
        if apparel_region.size == 0:
            return None

        hsv = cv2.cvtColor(apparel_region, cv2.COLOR_BGR2HSV)
        bins = max(4, int(hist_bins))
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm <= 0 or hist.size == 0:
            return None
        return hist / norm
    except Exception as e:
        logger.debug(f"提取直方图失败: {e}")
        return None


# ---------------------------------------------------------------------------
# 核心算法：与主流程完全一致的 Union-Find 聚类
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
    else:  # cosine
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
    与主流程 _build_cluster_mapping() + refine_*_with_clustering() 逻辑一致。

    Returns:
        key_to_cluster_id:  每个 key → 簇编号 (1-based, 按首次出现顺序)
        cluster_id_to_keys: 簇编号 → [key list]
        pair_details:       所有计算过的 pair 详情列表 (用于详细日志)
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
    metric: str = "cosine",
    threshold: float = 0.45,
    hist_bins: int = 8,
    min_cluster_size: int = 2,
    log_details: bool = False,
    feature_mode: str = "reid",
    reid_extractor: Optional[ReIDFeatureExtractor] = None,
) -> Dict[str, Any]:
    """
    对指定文件夹下的所有图片提取特征并聚类。
    feature_mode='reid' 优先使用 OSNet 深度特征，'hsv' 仅使用 HSV 直方图。
    """
    logger.info("=" * 70)
    logger.info("聚类任务开始")
    logger.info(f"  输入目录  : {os.path.abspath(image_dir)}")
    logger.info(f"  输出目录  : {os.path.abspath(output_dir)}")
    logger.info(f"  特征模式  : {feature_mode}")
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

    # 2. 提取特征
    logger.info(f"[步骤2] 提取特征 (mode={feature_mode}) ...")
    keys: List[str] = []
    embeddings: List[np.ndarray] = []
    failed: List[str] = []

    t0 = time.time()
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is None:
                failed.append(img_path)
                logger.debug(f"  无法读取: {img_path}")
                continue

            emb = extract_features(img, feature_mode=feature_mode, hist_bins=hist_bins, reid_extractor=reid_extractor)
            if emb is None:
                failed.append(img_path)
                logger.debug(f"  特征提取失败: {img_path}")
                continue

            keys.append(os.path.basename(img_path))
            embeddings.append(emb)
            logger.debug(f"  OK: {os.path.basename(img_path)} | emb_dim={emb.size}")
        except Exception as e:
            failed.append(img_path)
            logger.debug(f"  异常: {img_path} | {e}")

    t_extract = time.time() - t0
    logger.info(
        f"[步骤2] 完成: 成功={len(keys)}, 失败={len(failed)}, "
        f"耗时={t_extract:.2f}s"
    )

    if len(keys) < 2:
        logger.warning(f"有效图片不足 2 张 (仅 {len(keys)} 张)，无法聚类。")
        return {"status": "too_few", "valid_count": len(keys)}

    # 3. 计算原型向量（与主流程一致：每个 key 的所有 embedding 取均值）
    #    这里每个 key 只有一个 embedding，所以 proto 就是自身
    proto_vecs: List[np.ndarray] = []
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        if norm <= 0:
            proto_vecs.append(emb)
        else:
            proto_vecs.append((emb / norm).astype(np.float32))
    proto_mat = np.stack(proto_vecs, axis=0)
    logger.info(f"[步骤3] 原型向量矩阵: shape={proto_mat.shape}")

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

    logger.info("-" * 70)
    logger.info("[步骤5] 聚类结果统计:")
    logger.info(f"  总簇数       : {total_clusters}")
    logger.info(f"  有效簇 (≥{min_cluster_size}): {valid_clusters}")
    logger.info(f"  单例簇       : {singleton_clusters}")
    logger.info(f"  最大簇大小   : {max_cluster_size}")
    logger.info(f"  比较 pair 数 : {len(pair_details)}")
    merged_pairs = sum(1 for d in pair_details if d["matched"])
    logger.info(f"  合并 pair 数 : {merged_pairs}/{len(pair_details)}")
    logger.info("-" * 70)

    # 6. 输出每个簇的详情
    logger.info("[步骤6] 各簇详情:")
    for cid in sorted(cid_to_keys.keys()):
        members = cid_to_keys[cid]
        status = "✓ 有效" if len(members) >= min_cluster_size else "✗ 过小(单例)"
        logger.info(f"  Cluster #{cid}: {len(members)} 张 [{status}]")
        for k in sorted(members):
            logger.info(f"    - {k}")

    # 7. 输出比较对中 score 靠近阈值的边缘 case
    logger.info("[步骤7] 边缘 case (score 接近阈值):")
    if metric == "cosine":
        margin = 0.05
        edge_pairs = [
            d
            for d in pair_details
            if abs(d["score"] - threshold) < margin
        ]
    else:
        margin = threshold * 0.1
        edge_pairs = [
            d
            for d in pair_details
            if abs(d["score"] - threshold) < margin
        ]

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

    # 8. 写入输出文件
    os.makedirs(output_dir, exist_ok=True)

    # 8a. 聚类映射 CSV
    mapping_path = os.path.join(output_dir, "cluster_mapping.csv")
    with open(mapping_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "cluster_id", "cluster_size"])
        for k in keys:
            cid = key_to_cid[k]
            writer.writerow([k, cid, len(cid_to_keys[cid])])
    logger.info(f"[输出] 聚类映射: {mapping_path}")

    # 8b. 所有 pair 比较详情 CSV (用于调试)
    pairs_path = os.path.join(output_dir, "pair_details.csv")
    with open(pairs_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["key_i", "key_j", "score", "metric", "threshold", "matched"]
        )
        writer.writeheader()
        writer.writerows(pair_details)
    logger.info(f"[输出] Pair 详情: {pairs_path}")

    # 8c. 按簇统计摘要 CSV
    summary_path = os.path.join(output_dir, "cluster_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "size", "is_valid", "filenames"])
        for cid in sorted(cid_to_keys.keys()):
            members = cid_to_keys[cid]
            writer.writerow([
                cid,
                len(members),
                len(members) >= min_cluster_size,
                "; ".join(sorted(members)),
            ])
    logger.info(f"[输出] 簇摘要: {summary_path}")

    # 8d. 将图片按簇复制到子目录（方便可视化验证）
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
    logger.info("聚类任务完成!")
    logger.info("=" * 70)

    return {
        "status": "ok",
        "image_count": len(image_paths),
        "valid_count": len(keys),
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
    metric: str = "cosine",
    threshold: float = 0.45,
    hist_bins: int = 8,
    min_cluster_size: int = 2,
    log_details: bool = False,
    feature_mode: str = "reid",
    reid_extractor: Optional[ReIDFeatureExtractor] = None,
) -> Dict[str, Any]:
    """
    跨文件夹全局聚类。
    feature_mode='reid' 优先使用 OSNet 深度特征，'hsv' 仅使用 HSV 直方图。
    """
    logger.info("=" * 70)
    logger.info("跨文件夹全局聚类任务开始")
    logger.info(f"  根目录    : {os.path.abspath(output_root)}")
    logger.info(f"  特征模式  : {feature_mode}")
    logger.info(f"  相似度度量: {metric}")
    logger.info(f"  聚类阈值  : {threshold}")
    logger.info(f"  最小簇大小: {min_cluster_size}")
    logger.info("=" * 70)

    # 1. 扫描所有子目录
    subdirs: Dict[str, List[str]] = {}
    for entry in sorted(os.listdir(output_root)):
        entry_path = os.path.join(output_root, entry)
        if not os.path.isdir(entry_path):
            continue
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
    logger.info(f"[步骤2] 提取特征 (mode={feature_mode}) ...")
    source_to_embeddings: Dict[str, List[np.ndarray]] = {}
    source_meta: Dict[str, Dict[str, str]] = {}
    failed_total = 0
    t0 = time.time()

    for subdir, img_paths in subdirs.items():
        for img_path in img_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    failed_total += 1
                    continue

                emb = extract_features(img, feature_mode=feature_mode, hist_bins=hist_bins, reid_extractor=reid_extractor)
                if emb is None:
                    failed_total += 1
                    continue

                fname = os.path.basename(img_path)
                source_key = f"{subdir}/{fname}"
                source_to_embeddings.setdefault(source_key, []).append(hist)
                source_meta[source_key] = {"subdir": subdir, "filename": fname}
            except Exception:
                failed_total += 1

    t_extract = time.time() - t0
    total_valid = sum(len(v) for v in source_to_embeddings.values())
    logger.info(
        f"[步骤2] 完成: 有效={total_valid}, 失败={failed_total}, 耗时={t_extract:.2f}s"
    )

    if len(source_to_embeddings) < 2:
        logger.warning(f"有效图片不足 2 张，无法聚类。")
        return {"status": "too_few", "valid_count": total_valid}

    # 3. 原型向量
    keys = sorted(source_to_embeddings.keys())
    proto_vecs = []
    for k in keys:
        v = np.mean(np.stack(source_to_embeddings[k], axis=0), axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        proto_vecs.append(v.astype(np.float32))
    proto_mat = np.stack(proto_vecs, axis=0)
    logger.info(f"[步骤3] 原型矩阵 shape={proto_mat.shape}")

    # 4. 聚类
    logger.info(f"[步骤4] 全局 Union-Find 聚类 ...")
    t1 = time.time()
    key_to_cid, cid_to_keys, pair_details = union_find_clustering(
        keys, proto_mat, metric, threshold, min_cluster_size, log_details
    )
    t_cluster = time.time() - t1
    logger.info(f"[步骤4] 聚类完成, 耗时={t_cluster:.2f}s")

    # 5. 统计
    total_clusters = len(cid_to_keys)
    valid_clusters = sum(
        1 for m in cid_to_keys.values() if len(m) >= min_cluster_size
    )
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
    out_dir = os.path.join(output_root, "global_cluster_result")
    os.makedirs(out_dir, exist_ok=True)

    mapping_path = os.path.join(out_dir, "global_cluster_mapping.csv")
    with open(mapping_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_key", "subdir", "filename", "global_cluster_id", "cluster_size"])
        for k in keys:
            cid = key_to_cid[k]
            meta = source_meta.get(k, {})
            writer.writerow([
                k,
                meta.get("subdir", ""),
                meta.get("filename", ""),
                cid,
                len(cid_to_keys[cid]),
            ])
    logger.info(f"[输出] 全局映射: {mapping_path}")

    pairs_path = os.path.join(out_dir, "global_pair_details.csv")
    with open(pairs_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["key_i", "key_j", "score", "metric", "threshold", "matched"]
        )
        writer.writeheader()
        writer.writerows(pair_details)
    logger.info(f"[输出] Pair 详情: {pairs_path}")

    logger.info("=" * 70)
    logger.info("全局聚类任务完成!")
    logger.info("=" * 70)

    return {
        "status": "ok",
        "dir_count": len(subdirs),
        "valid_count": total_valid,
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
        description="独立的图片聚类测试工具（使用与主流程相同的 Union-Find 聚类逻辑）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单文件夹聚类
  python test/cluster_test.py ./output/video-3/half_body/ --threshold 0.45

  # 详细日志 + 输出日志文件
  python test/cluster_test.py ./output/video-3/half_body/ --verbose --log-file test/cluster.log

  # 跨文件夹全局聚类
  python test/cluster_test.py ./output/ --global --threshold 0.45
        """,
    )

    parser.add_argument("input_path", type=str, help="图片文件夹路径（单文件夹模式）或根目录路径（全局模式）")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="输出目录（默认: <input_path>_cluster_result）",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.45,
        help="聚类阈值（cosine模式: 0~1 越高越严格; euclidean模式: 越低越严格）",
    )
    parser.add_argument(
        "--metric", "-m",
        type=str,
        choices=["cosine", "euclidean"],
        default="cosine",
        help="相似度度量方式（默认: cosine）",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=8,
        help="HSV 直方图 bin 数量（默认: 8，即 8×8×8=512 维）",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="最小簇大小，低于此数的簇标记为无效（默认: 2）",
    )
    parser.add_argument(
        "--global-mode", "-g",
        action="store_true",
        dest="global_mode",
        help="启用跨文件夹全局聚类模式（扫描 input_path 下的所有子目录）",
    )
    parser.add_argument(        '--feature', '-f',
        type=str,
        choices=['reid', 'hsv'],
        default='reid',
        help='特征提取模式: reid (OSNet 深度特征, 推荐) / hsv (HSV 直方图)',
    )
    parser.add_argument(
        '--reid-model',
        type=str,
        default='models/reid/osnet_x0_25_msmt17.pt',
        help='OSNet ReID 模型路径（默认: models/reid/osnet_x0_25_msmt17.pt）',
    )
    parser.add_argument(        "--verbose", "-v",
        action="store_true",
        help="输出详细日志（包含每个 pair 的比较分数）",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="将日志同时写入指定文件",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input_path):
        logger.error(f"路径不存在: {args.input_path}")
        sys.exit(1)

    # 日志
    log_file = args.log_file
    setup_logging(verbose=args.verbose, log_file=log_file)

    if log_file:
        logger.info(f"日志文件: {os.path.abspath(log_file)}")

    # 初始化 ReID 提取器（如果需要）
    reid_extractor = None
    if args.feature == 'reid':
        model_path = args.reid_model
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        reid_extractor = ReIDFeatureExtractor(model_path=model_path)
        if not reid_extractor.is_available():
            logger.warning('ReID 模型不可用，自动回退到 HSV 直方图特征')
            args.feature = 'hsv'
            reid_extractor = None

    feature_label = 'ReID (OSNet)' if args.feature == 'reid' else 'HSV Histogram'
    logger.info(f'特征提取模式: {feature_label}')

    if args.global_mode:
        # 跨文件夹全局聚类
        result = cluster_global(
            output_root=args.input_path,
            metric=args.metric,
            threshold=args.threshold,
            hist_bins=args.hist_bins,
            min_cluster_size=args.min_cluster_size,
            log_details=args.verbose,
            feature_mode=args.feature,
            reid_extractor=reid_extractor,
        )
    else:
        # 单文件夹聚类
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = args.input_path.rstrip("/") + "_cluster_result"

        result = cluster_images_in_folder(
            image_dir=args.input_path,
            output_dir=output_dir,
            metric=args.metric,
            threshold=args.threshold,
            hist_bins=args.hist_bins,
            min_cluster_size=args.min_cluster_size,
            log_details=args.verbose,
            feature_mode=args.feature,
            reid_extractor=reid_extractor,
        )

    # 最终摘要
    logger.info("\n" + "▓" * 50)
    logger.info("最终摘要:")
    for k, v in result.items():
        logger.info(f"  {k}: {v}")
    logger.info("▓" * 50)


if __name__ == "__main__":
    main()
