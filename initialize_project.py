#!/usr/bin/env python3
"""
项目初始化脚本

功能:
  ✓ 验证模型完整性
  ✓ 验证项目结构
  ✓ 检查依赖
  ✓ 提供使用建议
  
建议在运行任何处理之前调用此脚本
"""

import os
import sys
import logging
from pathlib import Path

# 导入管理模块
from model_manager import ModelManager
from path_manager import PathManager
from args_manager import ArgumentManager

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """检查必需的Python依赖"""
    logger.info("\n检查Python依赖...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'onnxruntime': 'onnxruntime',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            logger.info(f"  ✓ {package_name}")
        except ImportError:
            logger.warning(f"  ✗ {package_name} 未安装")
            missing.append(package_name)
    
    if missing:
        logger.warning(f"\n⚠️  缺失依赖: {', '.join(missing)}")
        logger.info(f"请运行: pip install {' '.join(missing)}")
        return False
    
    return True


def check_models() -> bool:
    """检查模型"""
    logger.info("\n" + "="*60)
    logger.info("检查模型")
    logger.info("="*60)
    
    ModelManager.ensure_dirs_exist()
    results = ModelManager.verify_models(verbose=True)
    
    all_ok = all(results.values())
    
    if not all_ok:
        logger.warning("\n⚠️  某些模型缺失")
        logger.info("运行模型下载:")
        logger.info(f"  python download_insightface.py")
        logger.info(f"  python download_models.py")
    
    return all_ok


def check_project_structure() -> bool:
    """检查项目结构"""
    logger.info("\n" + "="*60)
    logger.info("检查项目结构")
    logger.info("="*60)
    
    required_files = [
        'config.py',
        'face_dedup_pipeline.py',
        'face_dedup_utils.py',
        'download_insightface.py',
        'requirements.txt',
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = Path(filename)
        if filepath.exists():
            logger.info(f"  ✓ {filename}")
        else:
            logger.warning(f"  ✗ {filename} 缺失")
            missing_files.append(filename)
    
    if missing_files:
        logger.error(f"\n✗ 项目文件缺失: {', '.join(missing_files)}")
        return False
    
    return True


def check_write_permissions() -> bool:
    """检查写入权限"""
    logger.info("\n检查写入权限...")
    
    test_dirs = [
        Path('models'),
        Path('output'),
        Path('logs'),
    ]
    
    all_ok = True
    for test_dir in test_dirs:
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / '.write_test'
        
        try:
            test_file.write_text('.')
            test_file.unlink()
            logger.info(f"  ✓ {test_dir}/")
        except Exception as e:
            logger.error(f"  ✗ {test_dir}/ ({e})")
            all_ok = False
    
    return all_ok


def check_command_line_args() -> bool:
    """检查参数系统"""
    logger.info("\n检查参数系统...")
    
    try:
        # 测试参数解析
        parser = ArgumentManager.create_parser()
        
        # 测试一个简单的参数列表
        test_args = parser.parse_args(['test_video.mp4', '--threshold', '0.6'])
        
        if test_args.threshold == 0.6:
            logger.info(f"  ✓ 参数解析工作正常")
            return True
        else:
            logger.error(f"  ✗ 参数解析失败")
            return False
    except Exception as e:
        logger.error(f"  ✗ 参数系统错误: {e}")
        return False


def print_quick_start() -> None:
    """打印快速开始指南"""
    logger.info("\n" + "="*60)
    logger.info("快速开始")
    logger.info("="*60)
    
    print("""
1️⃣  确保所有模型已下载:
    python download_insightface.py
    python download_models.py

2️⃣  基础用法:
    python face_dedup_pipeline.py videos/video.mp4

3️⃣  严格模式（更严格的正脸筛选）:
    python face_dedup_pipeline.py videos/video.mp4 --strict-mode

4️⃣  自定义参数:
        python face_dedup_pipeline.py videos/video.mp4 \
            --threshold 0.6 \
            --conf 0.6 \
      --yaw-threshold 20 \\
      --output-dir ./output/myfaces

5️⃣  使用配置文件:
    python face_dedup_pipeline.py videos/video.mp4 --config config.yaml

6️⃣  查看其他示例:
    python examples.py

📖  详细文档: 见 README.md

    """)


def main():
    """主初始化流程"""
    logger.info("="*60)
    logger.info("人脸检测模块 — 项目初始化")
    logger.info("="*60)
    
    checks = [
        ("依赖检查", check_dependencies),
        ("项目结构", check_project_structure),
        ("写入权限", check_write_permissions),
        ("模型检查", check_models),
        ("参数系统", check_command_line_args),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            logger.error(f"检查失败: {e}")
            results[check_name] = False
    
    # 摘要
    logger.info("\n" + "="*60)
    logger.info("初始化摘要")
    logger.info("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✓ 通过" if result else "✗ 未通过"
        logger.info(f"{status}: {check_name}")
    
    logger.info(f"\n总体: {passed}/{total} 项检查通过")
    
    if passed == total:
        logger.info("\n✅ 所有检查通过，项目可用！")
        print_quick_start()
        return 0
    else:
        logger.error("\n❌ 某些检查未通过，请解决问题后重试")
        return 1


if __name__ == '__main__':
    sys.exit(main())
