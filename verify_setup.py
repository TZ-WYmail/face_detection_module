#!/usr/bin/env python3
"""
快速验证脚本 - 检查模型下载和 tracking 功能是否正常

用法：
    python verify_setup.py         # 完整检查
    python verify_setup.py --quick # 快速检查（不下载）
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_file_exists(path, name=""):
    """检查文件是否存在"""
    if Path(path).exists():
        size = Path(path).stat().st_size / (1024 * 1024) if Path(path).is_file() else 0
        size_str = f" ({size:.1f} MB)" if size > 0 else ""
        print(f"  ✅ 存在: {path}{size_str}")
        return True
    else:
        print(f"  ❌ 缺失: {path}")
        return False


def check_directory_py_syntax(path):
    """检查 Python 文件的语法"""
    if not Path(path).exists():
        print(f"  ❌ 文件不存在: {path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", path],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"  ✅ 语法检查通过: {path}")
            return True
        else:
            print(f"  ❌ 语法错误: {path}")
            if result.stderr:
                print(f"     错误: {result.stderr.decode('utf-8', errors='replace')}")
            return False
    except Exception as e:
        print(f"  ❌ 无法检查: {path} ({e})")
        return False


def main():
    parser = argparse.ArgumentParser(description="快速验证项目设置")
    parser.add_argument("--quick", action="store_true", help="快速检查（不下载）")
    parser.add_argument("--download", action="store_true", help="执行模型下载")
    args = parser.parse_args()

    print_header("🔍 人脸检测项目设置验证")

    all_ok = True

    # 1. 检查主要Python文件
    print_header("1️⃣  检查核心文件")
    
    core_files = [
        "download_models.py",
        "face_dedup_pipeline.py",
        "face_dedup_utils.py",
        "config.py",
    ]
    
    for file in core_files:
        if not check_directory_py_syntax(file):
            all_ok = False

    # 2. 检查目录结构
    print_header("2️⃣  检查目录结构")
    
    directories = [
        "models/",
        "models/yolo/",
        "models/insightface/",
        "output/",
        "logs/",
        "data/",
    ]
    
    for dir_path in directories:
        if Path(dir_path).exists():
            print(f"  ✅ 存在: {dir_path}")
        else:
            print(f"  ⚠️  缺失: {dir_path} (将在运行时创建)")

    # 3. 检查模型文件
    print_header("3️⃣  检查模型文件")
    
    yolo_pt_models = [
        "models/yolo/yolov11n-face.pt",
        "models/yolo/yolov11s-face.pt",
        "models/yolo/yolov11m-face.pt",
        "models/yolo/yolov11l-face.pt",
        "models/yolo/yolo26n-face.pt",
        "models/yolo/yolov10n-face.pt",
    ]
    
    found_pt = False
    print("  检查 YOLO .pt 模型（支持 tracking）：")
    for model_path in yolo_pt_models:
        if Path(model_path).exists():
            size = Path(model_path).stat().st_size / (1024 * 1024)
            print(f"    ✅ {model_path} ({size:.1f} MB)")
            found_pt = True
    
    if not found_pt:
        print("    ⚠️  未找到 YOLO .pt 模型")
        print(f"    💡 提示: 运行以下命令下载:")
        print(f"       python download_models.py --all-yolo")
        all_ok = False
    
    # 检查 InsightFace 模型
    print("\n  检查 InsightFace 模型：")
    insightface_models = [
        "models/insightface/models/buffalo_l/",
    ]
    
    found_insightface = False
    for model_dir in insightface_models:
        if Path(model_dir).exists():
            onnx_files = list(Path(model_dir).rglob("*.onnx"))
            if onnx_files:
                print(f"    ✅ {model_dir}")
                for f in onnx_files:
                    size = f.stat().st_size / (1024 * 1024)
                    print(f"       - {f.name} ({size:.1f} MB)")
                found_insightface = True
    
    if not found_insightface:
        print(f"    ⚠️  未找到 InsightFace 模型")
        print(f"    💡 提示: 运行以下命令下载:")
        print(f"       python download_models.py")

    # 4. 导入检查
    print_header("4️⃣  检查依赖库")
    
    dependencies = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("ultralytics", "Ultralytics YOLO"),
        ("torch", "PyTorch"),
        ("insightface", "InsightFace"),
        ("huggingface_hub", "Hugging Face Hub"),
    ]
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ⚠️  {display_name} 未安装")

    # 5. 推荐的后续步骤
    print_header("📋 推荐的后续步骤")
    
    if not found_pt:
        print("1️⃣  下载 YOLO .pt 模型（支持 tracking）：")
        print("    python download_models.py --all-yolo")
        print()
    
    if not found_insightface:
        print("2️⃣  下载 InsightFace 模型（人脸识别）：")
        print("    python download_models.py")
        print()
    
    print("3️⃣  启用 ByteTrack 跟踪处理视频：")
    print("    python face_dedup_pipeline.py --video input.mp4 --use-tracks")
    print()
    
    print("4️⃣  查看更多选项：")
    print("    python download_models.py --help")
    print("    python face_dedup_pipeline.py --help")

    # 6. 最终状态
    print_header("✅ 验证摘要")
    
    if found_pt and found_insightface:
        print("🎉 所有关键模型已安装！可以开始使用了。")
        print("\n✨ 快速开始：")
        print("   python face_dedup_pipeline.py --video sample.mp4 --use-tracks")
    elif found_insightface:
        print("⚠️  缺少 YOLO .pt 模型（需要 tracking 功能）")
        print("\n💡 执行以下命令下载模型：")
        print("   python download_models.py --all-yolo")
    else:
        print("⚠️  缺少核心模型")
        print("\n💡 执行以下命令下载所有模型：")
        print("   python download_models.py --all-yolo")
    
    print()

    return 0 if (found_pt and found_insightface) else 1


if __name__ == "__main__":
    sys.exit(main())
