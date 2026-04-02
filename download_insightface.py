#!/usr/bin/env python3
"""
InsightFace 完整模型下载脚本

下载 Buffalo_L 全套 5 个子模型（ONNX）：
  1. det_10g.onnx    — 人脸检测（RetinaFace，支持 10G 显存）
  2. w600k_r50.onnx  — 人脸特征提取（ArcFace，512维）
  3. 2d106det.onnx   — 106点人脸关键点
  4. 1k3d68.onnx     — 68点 3D 人脸关键点（用于姿态估计）
  5. genderage.onnx  — 性别与年龄估计

下载源（按优先级依次尝试）：
  方式1: insightface 官方包内置下载（最稳定）
  方式2: HuggingFace immich-app/buffalo_l（公开，无需 Token）
  方式3: HuggingFace 直链逐个下载

用法：
    python download_insightface.py
    python download_insightface.py --insightface-dir ./models/insightface
    python download_insightface.py --skip-detection   # 跳过检测模型（如果你用 YOLO）
    python download_insightface.py --verify-only      # 只验证，不下载
"""

from __future__ import annotations

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================================================
#  Buffalo_L 完整子模型清单
# ======================================================================
BUFFALO_L_FILES = {
    "det_10g.onnx": {
        "desc": "人脸检测 (RetinaFace, 10G显存优化)",
        "pipeline_step": "步骤2: 人脸检测",
        "used_by_you": "可跳过 — 你用 YOLO 替代",
        "size_mb": 16.1,
    },
    "w600k_r50.onnx": {
        "desc": "人脸特征提取 (ArcFace, 512维 Embedding)",
        "pipeline_step": "步骤7: Embedding提取",
        "used_by_you": "★ 必须使用 — 提取 512维特征向量",
        "size_mb": 166.3,
    },
    "2d106det.onnx": {
        "desc": "106点 2D 人脸关键点检测",
        "pipeline_step": "步骤4: 关键点提取",
        "used_by_you": "可选 — 高精度关键点对齐",
        "size_mb": 12.3,
    },
    "1k3d68.onnx": {
        "desc": "68点 3D 人脸关键点检测",
        "pipeline_step": "步骤5: 姿态估计",
        "used_by_you": "★ 推荐 — 用于 yaw/pitch/roll 计算",
        "size_mb": 30.2,
    },
    "genderage.onnx": {
        "desc": "性别与年龄估计",
        "pipeline_step": "步骤6: 质量评估 (辅助)",
        "used_by_you": "可选 — 辅助筛选",
        "size_mb": 3.5,
    },
}

HF_REPO = "immich-app/buffalo_l"


# ======================================================================
#  方式1: insightface 官方包内置下载
# ======================================================================
def download_via_insightface(target_dir: Path, skip_detection: bool = False) -> bool:
    """
    利用 insightface.app.FaceAnalysis 触发官方模型下载。
    这是最稳定的方式，会自动下载到 insightface 默认缓存目录。
    """
    logger.info("\n" + "=" * 60)
    logger.info("方式1: 通过 insightface 官方包下载")
    logger.info("=" * 60)

    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        logger.info("请安装: pip install insightface onnxruntime")
        return False

    default_model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    logger.info(f"insightface 默认模型目录: {default_model_dir}")

    try:
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(Path.home() / ".insightface" / "models"),
            allowed_modules=["detection", "recognition",
                             "landmark_2d_106", "landmark_3d_68", "genderage"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("✅ insightface 官方下载成功")

        # 复制到目标目录（如果用户指定了不同目录）
        if target_dir.resolve() != default_model_dir.resolve():
            logger.info(f"复制到目标目录: {target_dir}")
            target_dir.mkdir(parents=True, exist_ok=True)
            for f in default_model_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, target_dir / f.name)
                    logger.info(f"   复制: {f.name}")
        return True

    except Exception as e:
        logger.warning(f"方式1 失败: {e}")
        return False


# ======================================================================
#  方式2: HuggingFace snapshot_download
# ======================================================================
def download_via_huggingface(target_dir: Path) -> bool:
    """
    从 HuggingFace 公开仓库 immich-app/buffalo_l 下载。
    """
    logger.info("\n" + "=" * 60)
    logger.info("方式2: 通过 HuggingFace 下载")
    logger.info("=" * 60)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("缺少 huggingface_hub: pip install huggingface_hub")
        return False

    token = os.environ.get("HF_TOKEN") or None

    try:
        logger.info(f"仓库: {HF_REPO} (公开，无需Token)")
        logger.info(f"目标: {target_dir}")

        path = snapshot_download(
            repo_id=HF_REPO,
            local_dir=str(target_dir),
            token=token,
        )
        logger.info(f"✅ HuggingFace 下载完成: {path}")
        return True

    except Exception as e:
        logger.warning(f"方式2 失败: {e}")
        return False


# ======================================================================
#  方式3: HuggingFace 直链逐个下载 ONNX
# ======================================================================
def download_via_direct_url(target_dir: Path) -> bool:
    """
    从 HuggingFace 文件直链逐个下载 ONNX。
    """
    logger.info("\n" + "=" * 60)
    logger.info("方式3: 从 HuggingFace 直链逐个下载")
    logger.info("=" * 60)

    import urllib.request

    target_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    file_map = {
        "det_10g.onnx":    "detection/model.onnx",
        "w600k_r50.onnx":  "recognition/model.onnx",
        # 使用指定的直链以提高可用性
        "2d106det.onnx":   "https://huggingface.co/lithiumice/insightface/resolve/main/models/buffalo_l/2d106det.onnx",
        "1k3d68.onnx":     "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/1k3d68.onnx",
        "genderage.onnx":  "genderage/model.onnx",
    }

    for target_name, repo_path in file_map.items():
        # 如果 repo_path 是完整 URL 则直接使用，否则按原有 HF_REPO 构造地址
        if isinstance(repo_path, str) and repo_path.startswith("http"):
            url = repo_path
        else:
            url = f"https://huggingface.co/{HF_REPO}/resolve/main/{repo_path}"
        save_path = target_dir / target_name

        if save_path.exists() and save_path.stat().st_size > 0:
            logger.info(f"✅ 已存在: {target_name}")
            success_count += 1
            continue

        logger.info(f"⬇  {target_name} ← {repo_path}")
        try:
            req = urllib.request.Request(url)
            if os.environ.get("HF_TOKEN"):
                req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")

            with urllib.request.urlopen(req, timeout=120) as resp, \
                 open(save_path, "wb") as f:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = min(downloaded * 100 / total, 100)
                        filled = int(50 * pct // 100)
                        bar = "█" * filled + "░" * (50 - filled)
                        print(f"\r  |{bar}| {pct:5.1f}%", end="", flush=True)
                    else:
                        print(f"\r  {downloaded/1024/1024:.1f} MB", end="", flush=True)
            print()

            if save_path.stat().st_size > 0:
                size = save_path.stat().st_size / (1024 * 1024)
                logger.info(f"  ✅ {target_name} ({size:.1f} MB)")
                success_count += 1
            else:
                save_path.unlink(missing_ok=True)
                logger.warning(f"  ❌ {target_name} 下载为空")
        except Exception as e:
            save_path.unlink(missing_ok=True)
            logger.warning(f"  ❌ {target_name} 失败: {e}")

    logger.info(f"方式3 结果: {success_count}/{len(file_map)} 个文件成功")
    return success_count == len(file_map)


# ======================================================================
#  验证模型完整性
# ======================================================================
def verify_models(model_dir: Path, skip_detection: bool = False) -> dict[str, bool]:
    """验证所有 ONNX 文件是否就位，并打印流水线对应关系。"""
    logger.info("\n" + "=" * 60)
    logger.info("模型验证")
    logger.info("=" * 60)

    results = {}
    total_size = 0
    all_onnx = {f.name: f for f in model_dir.rglob("*.onnx")
                if ".cache" not in str(f)}

    name_aliases = {
        "det_10g.onnx":   ["det_10g.onnx", "model.onnx"],
        "w600k_r50.onnx": ["w600k_r50.onnx", "model.onnx"],
        "2d106det.onnx":  ["2d106det.onnx"],
        "1k3d68.onnx":    ["1k3d68.onnx"],
        "genderage.onnx": ["genderage.onnx", "model.onnx"],
    }

    for std_name, info in BUFFALO_L_FILES.items():
        if skip_detection and std_name == "det_10g.onnx":
            logger.info(f"  ⏭️  {std_name:20s} — 已跳过（使用 YOLO 检测）")
            results[std_name] = True
            continue

        found = False
        fpath = None
        if std_name in all_onnx:
            found = True
            fpath = all_onnx[std_name]
        else:
            aliases = name_aliases.get(std_name, [std_name])
            for fp in all_onnx.values():
                if any(alias in str(fp) for alias in aliases):
                    found = True
                    fpath = fp
                    break

        if found and fpath:
            size_mb = fpath.stat().st_size / (1024 * 1024)
            total_size += fpath.stat().st_size
            results[std_name] = True
            logger.info(f"  ✅ {std_name:20s} {size_mb:7.1f} MB  {info['desc']}")
        else:
            results[std_name] = False
            logger.info(f"  ❌ {std_name:20s} {'—':>7s}     {info['desc']}")

    # 流水线映射
    logger.info(f"\n{'='*60}")
    logger.info("流水线中各模型用途")
    logger.info(f"{'='*60}")
    for std_name, info in BUFFALO_L_FILES.items():
        logger.info(f"  {info['used_by_you']}")
        logger.info(f"      {std_name:20s}  →  {info['pipeline_step']}")

    ok_count = sum(results.values())
    total_count = len(results)
    total_mb = total_size / (1024 * 1024)
    logger.info(f"\n总计: {ok_count}/{total_count} 个模型就位, {total_mb:.1f} MB")

    return results


# ======================================================================
#  搜索系统中已有的 insightface 模型
# ======================================================================
def find_existing_models() -> list[Path]:
    candidates = []
    search_paths = [
        Path.home() / ".insightface" / "models",
        Path(".") / "models",
        Path(".") / "models" / "insightface",
        Path(".") / "models" / "insightface" / "models",
    ]
    for p in search_paths:
        if p.exists():
            for sub in p.iterdir():
                if sub.is_dir():
                    onnx_count = len(list(sub.rglob("*.onnx")))
                    if onnx_count > 0:
                        candidates.append(sub)
    return candidates


# ======================================================================
#  主函数
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="InsightFace Buffalo_L 完整模型下载",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_insightface.py
  python download_insightface.py --insightface-dir ./models/insightface
  python download_insightface.py --skip-detection
  python download_insightface.py --verify-only
  python download_insightface.py --method huggingface
  python download_insightface.py --method direct
""",
    )
    parser.add_argument(
        "--insightface-dir", type=str, default=None,
        help="模型保存目录 (默认: ./models/insightface/buffalo_l)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "insightface", "huggingface", "direct"],
        default="auto",
        help="下载方式 (默认: auto 依次尝试)",
    )
    parser.add_argument(
        "--skip-detection", action="store_true",
        help="跳过 det_10g.onnx（你用 YOLO 做检测时推荐）",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="只验证，不下载",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.insightface_dir:
        target_dir = Path(args.insightface_dir)
    else:
        target_dir = Path("./models/insightface/buffalo_l")
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"模型目录: {target_dir.resolve()}")

    # 搜索已有模型
    existing = find_existing_models()
    if existing:
        logger.info("发现已有的 InsightFace 模型:")
        for d in existing:
            onnx_files = list(d.rglob("*.onnx"))
            total_mb = sum(f.stat().st_size for f in onnx_files) / (1024*1024)
            logger.info(f"  {d}  ({len(onnx_files)} ONNX, {total_mb:.1f} MB)")

    if args.verify_only:
        results = verify_models(target_dir, args.skip_detection)
        sys.exit(0 if all(results.values()) else 1)

    # ---- 依次尝试下载 ----
    ok = False
    if args.method in ("auto", "insightface"):
        ok = download_via_insightface(target_dir, args.skip_detection)
    if not ok and args.method in ("auto", "huggingface"):
        ok = download_via_huggingface(target_dir)
    if not ok and args.method in ("auto", "direct"):
        ok = download_via_direct_url(target_dir)
    if not ok and args.method == "insightface":
        ok = download_via_huggingface(target_dir)

    if not ok:
        logger.error("\n所有下载方式均失败")
        logger.info("手动下载: https://huggingface.co/immich-app/buffalo_l")
        sys.exit(1)

    # ---- 验证 ----
    results = verify_models(target_dir, args.skip_detection)

    if all(results.values()):
        logger.info(f"\n✅ InsightFace 所有模型准备完成！")
        logger.info(f"   位置: {target_dir.resolve()}")
        logger.info("\n使用示例:")
        logger.info("  from insightface.app import FaceAnalysis")
        logger.info("  app = FaceAnalysis(name='buffalo_l', root='./models/insightface')")
        logger.info("  app.prepare(ctx_id=0, det_size=(640, 640))")
    else:
        missing = [k for k, v in results.items() if not v]
        logger.warning(f"\n缺失模型: {missing}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
